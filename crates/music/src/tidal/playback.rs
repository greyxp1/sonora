//! Playback for Tidal: what [`crate::engine`] needs that is Tidal's own.
//!
//! A lossless track is one FLAC file and plays like any other. A hi-res track is a DASH stream
//! of fMP4 segments, which [`Parts`] fetches one after another into one stream, so to the
//! decoder it is one fragmented file. The manifest says when every segment starts and
//! [`Timeline`] notes where each one lands in the stream, so a seek splices the init segment
//! onto the segment that covers the position instead of decoding its way there.

use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use async_trait::async_trait;
use bytes::Bytes;
use reqwest::StatusCode;
use rodio::Source as _;

use crate::audio::Trimmed;
use crate::engine::{self, Fetch, Loudness};
use crate::stream::{self, Reader, Stream};
use crate::tidal::client::TidalClient;
use crate::tidal::lock;
use crate::tidal::manifest::Manifest;
use crate::{PlaybackConfig, PlaybackEvents, PlaybackFactory, Player, Quality};

/// How long one part keeps being asked for, with the wait between asks doubling from
/// `FIRST_WAIT` up to `LONGEST_WAIT`. A track is a hundred or so requests, so one that fails
/// now and then is ordinary, and a connection that drops for half a minute comes back to the
/// same track.
const RETRY_WINDOW: Duration = Duration::from_secs(45);
const FIRST_WAIT: Duration = Duration::from_millis(500);
const LONGEST_WAIT: Duration = Duration::from_secs(8);

/// How many times one track asks Tidal for fresh urls when the signed ones it holds are refused.
const RENEWALS: u32 = 3;

/// How long a media request may take to connect, and how long it may wait between two reads.
/// Both sit well under the patience of a reader waiting at the end of the download, so a stalled
/// request is asked again while that reader still waits.
const MEDIA_CONNECT: Duration = Duration::from_secs(5);
const MEDIA_STALL: Duration = Duration::from_secs(8);

/// How long a track may take to arrive as far as the preroll before its load fails.
const PRIMING: Duration = Duration::from_secs(20);

/// The most one segment may hold. A hi-res segment is a megabyte or two, so this only bounds
/// what a wrong answer can cost.
const SEGMENT_CEILING: usize = 64 * 1024 * 1024;

/// The most one Tidal track may spool. 24-bit FLAC at 192 kHz runs to about 40 MB a minute, so
/// this holds more than an hour and a half of it. On a 32-bit target it is as much as a size
/// can count.
const SPOOL_CEILING: u64 = 4 * 1024 * 1024 * 1024;

/// A decoder over a Tidal track, with what comes before the asked-for position trimmed off.
type Opened = Trimmed<rodio::Decoder<Reader<Long>>>;

/// Where the segments of a segmented track start, in the music and in the stream.
struct Timeline {
    /// When each media segment starts, from the manifest.
    starts: Vec<Duration>,
    /// Where each part starts in the stream: the init segment first, then the media segments.
    /// It grows as the parts arrive.
    offsets: Vec<usize>,
}

/// Where to start reading to land on a position, and how much of the first segment to throw
/// away to land exactly on it.
#[derive(Clone, Copy, Debug)]
struct Entry {
    /// The end of the init segment, which is spliced in front of the segment.
    head: usize,
    /// Where the segment starts in the stream.
    graft: usize,
    /// How far into that segment the position is.
    into: Duration,
}

/// The init segment and media segments of a DASH track, fetched one after another and handed
/// to the stream as one body. Each segment is fetched whole before any of it is handed over, so
/// a connection that drops part way is asked again rather than ending the track. It writes where
/// each part starts into the timeline, and stops once nothing holds the timeline, which is how
/// an abandoned track stops downloading.
struct Parts {
    http: reqwest::Client,
    /// The init segment's url, then the media segments' in order.
    urls: Vec<String>,
    /// The index in `urls` of the part to fetch next.
    next: usize,
    /// How many bytes have been handed to the stream, which is where the next part starts.
    sent: usize,
    timeline: Weak<Mutex<Timeline>>,
    renewal: Renewal,
}

/// A track that is one file, read as one response and asked for again from the first missing
/// byte when the connection drops. A refused url is renewed the way a segment's is.
struct Resuming {
    http: reqwest::Client,
    url: String,
    /// The response being read. `None` between a drop and the request that resumes it.
    response: Option<reqwest::Response>,
    /// How many bytes have been handed to the stream, which is where a resumed request starts.
    received: u64,
    /// The file's length, when the server said.
    total: Option<u64>,
    renewal: Renewal,
}

/// What a download needs to ask Tidal for fresh urls once the signed ones it holds are refused.
struct Renewal {
    client: TidalClient,
    id: String,
    /// The codec the download started in, which fresh urls have to name too.
    codec: String,
    /// The level the download was asked for at, which fresh urls are asked for at too.
    quality: Quality,
    /// How many more times the urls may be renewed.
    left: u32,
}

/// The waits between asks for one thing, and when the asking stops.
struct Backoff {
    since: Instant,
    wait: Duration,
}

/// Why a media request failed, which decides what is tried next.
enum Failure {
    /// The signed url was refused or has gone, so Tidal is asked for fresh ones.
    Expired(anyhow::Error),
    /// The connection or the server failed, which waiting may cure.
    Passing(anyhow::Error),
    /// Asking again would get the same answer.
    Final(anyhow::Error),
}

/// The body of a Tidal track, taken as it comes but allowed to run as long as hi-res does.
struct Long;

/// A track arriving, and how long and how loud Tidal says it is.
#[derive(Clone)]
pub struct Loaded {
    stream: Stream<Long>,
    /// Where a segmented track's segments start. `None` for a track that is one file.
    timeline: Option<Arc<Mutex<Timeline>>>,
    duration: Option<Duration>,
    loudness: Option<Loudness>,
}

/// Starts a playback engine for a signed-in session, which is what a [`crate::ProviderSession`]
/// hands the app.
pub struct Factory {
    client: TidalClient,
    media: reqwest::Client,
}

/// What the shared engine is given: how to fetch a Tidal track and open a decoder over it.
struct Tidal {
    client: TidalClient,
    /// The http client the track's bytes come through, which gives up on a stalled request
    /// sooner than the api client does.
    media: reqwest::Client,
    /// The level every track is asked for at.
    quality: Quality,
}

impl Timeline {
    /// The segment covering `at`, once the download has reached it.
    fn entry(&self, at: Duration) -> Option<Entry> {
        let index = self
            .starts
            .partition_point(|start| *start <= at)
            .checked_sub(1)?;
        Some(Entry {
            head: *self.offsets.get(1)?,
            graft: *self.offsets.get(index + 1)?,
            into: at.saturating_sub(self.starts[index]),
        })
    }
}

impl Parts {
    /// Fetches the next part whole. A failed connection or server is asked again with a growing
    /// wait, and a refused url is renewed before the same part is asked for again. `None` once
    /// every part is in or the track was abandoned, which is checked before every attempt.
    async fn fetch(&mut self) -> Result<Option<Bytes>> {
        let mut backoff = Backoff::new();
        loop {
            if self.timeline.strong_count() == 0 {
                return Ok(None);
            }
            let Some(url) = self.urls.get(self.next) else {
                return Ok(None);
            };
            match whole(&self.http, url).await {
                Ok(part) => return Ok(Some(part)),
                Err(Failure::Expired(error)) => {
                    log::debug!("tidal: segment {} was refused: {error:#}", self.next);
                    self.renew().await?;
                }
                Err(Failure::Passing(error)) => backoff.wait(error).await?,
                Err(Failure::Final(error)) => return Err(error),
            }
        }
    }

    /// Swaps in fresh urls for the same segments, so the download carries on from the part it
    /// was on.
    async fn renew(&mut self) -> Result<()> {
        let Manifest::Segments { init, media, .. } = self.renewal.manifest().await? else {
            bail!("tidal no longer streams the track as segments");
        };
        if media.len() + 1 != self.urls.len() {
            bail!(
                "tidal now cuts the track into {} segments rather than {}",
                media.len(),
                self.urls.len() - 1
            );
        }
        self.urls = std::iter::once(init).chain(media).collect();
        Ok(())
    }
}

impl stream::Source for Parts {
    async fn chunk(&mut self) -> Result<Option<Bytes>> {
        let Some(part) = self
            .fetch()
            .await
            .context("cannot fetch a segment of the tidal track")?
        else {
            return Ok(None);
        };
        let Some(timeline) = self.timeline.upgrade() else {
            return Ok(None);
        };
        lock(&timeline).offsets.push(self.sent);
        self.sent += part.len();
        self.next += 1;
        Ok(Some(part))
    }
}

impl Resuming {
    /// Makes the first request, asking again the way a dropped connection would.
    async fn open(http: reqwest::Client, url: String, renewal: Renewal) -> Result<Self> {
        let mut file = Self {
            http,
            url,
            response: None,
            received: 0,
            total: None,
            renewal,
        };
        file.connect()
            .await
            .context("cannot stream the tidal track")?;
        file.total = file
            .response
            .as_ref()
            .and_then(reqwest::Response::content_length);
        Ok(file)
    }

    /// Opens a response from the first byte not yet received. A failed connection or server is
    /// asked again with a growing wait, and a refused url is renewed first.
    async fn connect(&mut self) -> Result<()> {
        let mut backoff = Backoff::new();
        loop {
            match self.request().await {
                Ok(response) => {
                    self.response = Some(response);
                    return Ok(());
                }
                Err(Failure::Expired(error)) => {
                    log::debug!("tidal: the file was refused: {error:#}");
                    self.renew().await?;
                }
                Err(Failure::Passing(error)) => backoff.wait(error).await?,
                Err(Failure::Final(error)) => return Err(error),
            }
        }
    }

    async fn renew(&mut self) -> Result<()> {
        let Manifest::File { url, .. } = self.renewal.manifest().await? else {
            bail!("tidal no longer streams the track as one file");
        };
        self.url = url;
        Ok(())
    }

    /// One request from the first byte not yet received. A resumed request has to be answered
    /// with exactly the rest of the same file, or the track would splice two files together.
    async fn request(&self) -> Result<reqwest::Response, Failure> {
        let mut request = self.http.get(&self.url);
        if self.received > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={}-", self.received));
        }
        let response = Failure::check(request.send().await?)?;
        if self.received == 0 {
            return Ok(response);
        }
        if response.status() != StatusCode::PARTIAL_CONTENT {
            return Err(Failure::Final(anyhow!(
                "the tidal file server cannot resume a download"
            )));
        }
        let lines_up = content_range(&response).is_some_and(|(start, total)| {
            start == self.received && (self.total.is_none() || total == self.total)
        });
        match lines_up {
            true => Ok(response),
            false => Err(Failure::Final(anyhow!(
                "the tidal file server resumed somewhere other than byte {}",
                self.received
            ))),
        }
    }
}

impl stream::Source for Resuming {
    async fn chunk(&mut self) -> Result<Option<Bytes>> {
        let mut backoff = Backoff::new();
        loop {
            let Some(response) = &mut self.response else {
                self.connect()
                    .await
                    .context("cannot resume the tidal track")?;
                continue;
            };
            let error = match response.chunk().await {
                Ok(Some(chunk)) => {
                    self.received += chunk.len() as u64;
                    return Ok(Some(chunk));
                }
                Ok(None) => match self.total {
                    Some(total) if self.received < total => {
                        anyhow!("the tidal file ended {} bytes early", total - self.received)
                    }
                    _ => return Ok(None),
                },
                Err(error) => error.into(),
            };
            self.response = None;
            backoff
                .wait(error)
                .await
                .context("cannot stream the tidal track")?;
        }
    }
}

impl Renewal {
    /// A fresh manifest for the same track in the same codec. It fails once `RENEWALS` are
    /// spent, since a refusal that outlasts them is not an expired url.
    async fn manifest(&mut self) -> Result<Manifest> {
        if self.left == 0 {
            bail!("tidal keeps refusing the urls of {}", self.id);
        }
        self.left -= 1;
        log::info!("tidal: asking for fresh urls for {}", self.id);
        let manifest = self.client.playback(&self.id, self.quality).await?.manifest;
        if manifest.codec() != self.codec {
            bail!(
                "tidal now streams {} as {} rather than {}",
                self.id,
                manifest.codec(),
                self.codec
            );
        }
        Ok(manifest)
    }
}

impl Backoff {
    fn new() -> Self {
        Self {
            since: Instant::now(),
            wait: FIRST_WAIT,
        }
    }

    /// Waits before the next ask, or hands `error` back once `RETRY_WINDOW` has passed since
    /// the first one.
    async fn wait(&mut self, error: anyhow::Error) -> Result<()> {
        if self.since.elapsed() >= RETRY_WINDOW {
            return Err(error);
        }
        log::debug!("tidal: asking again in {:?} after {error:#}", self.wait);
        tokio::time::sleep(self.wait).await;
        self.wait = (self.wait * 2).min(LONGEST_WAIT);
        Ok(())
    }
}

impl Failure {
    /// Sorts a response by its status, passing a successful one through.
    fn check(response: reqwest::Response) -> Result<reqwest::Response, Self> {
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let error = anyhow!("the tidal media server answered {status}");
        Err(match status {
            StatusCode::FORBIDDEN | StatusCode::GONE => Self::Expired(error),
            StatusCode::REQUEST_TIMEOUT | StatusCode::TOO_MANY_REQUESTS => Self::Passing(error),
            status if status.is_server_error() => Self::Passing(error),
            _ => Self::Final(error),
        })
    }
}

impl From<reqwest::Error> for Failure {
    /// A request that could not be built fails the same way every time. Anything else went
    /// wrong on the way and may not again.
    fn from(error: reqwest::Error) -> Self {
        match error.is_builder() {
            true => Self::Final(error.into()),
            false => Self::Passing(error.into()),
        }
    }
}

impl stream::Body for Long {
    fn ceiling(&self) -> usize {
        usize::try_from(SPOOL_CEILING).unwrap_or(usize::MAX)
    }
}

impl Factory {
    pub fn new(client: TidalClient) -> Self {
        let media = media_http().unwrap_or_else(|error| {
            log::warn!("tidal: {error:#}, streaming through the api client");
            client.http().clone()
        });
        Self { client, media }
    }
}

impl PlaybackFactory for Factory {
    fn start(&self, config: PlaybackConfig) -> (Box<dyn Player>, Box<dyn PlaybackEvents>) {
        engine::start(
            Tidal {
                client: self.client.clone(),
                media: self.media.clone(),
                quality: config.quality,
            },
            config,
        )
    }
}

impl Tidal {
    /// Starts the download a manifest describes and waits for the preroll.
    async fn arrive(
        &self,
        id: &str,
        manifest: Manifest,
    ) -> Result<(Stream<Long>, Option<Arc<Mutex<Timeline>>>)> {
        let renewal = Renewal {
            client: self.client.clone(),
            id: id.to_owned(),
            codec: manifest.codec().to_owned(),
            quality: self.quality,
            left: RENEWALS,
        };
        let (stream, timeline) = match manifest {
            Manifest::File { url, .. } => {
                let file = Resuming::open(self.media.clone(), url, renewal).await?;
                let total = file.total;
                (Stream::pulling(file, total, Long), None)
            }
            Manifest::Segments {
                init,
                media,
                starts,
                ..
            } => {
                let timeline = Arc::new(Mutex::new(Timeline {
                    starts,
                    offsets: Vec::new(),
                }));
                let parts = Parts {
                    http: self.media.clone(),
                    urls: std::iter::once(init).chain(media).collect(),
                    next: 0,
                    sent: 0,
                    timeline: Arc::downgrade(&timeline),
                    renewal,
                };
                (Stream::pulling(parts, None, Long), Some(timeline))
            }
        };
        Ok((stream.primed().await?, timeline))
    }
}

#[async_trait]
impl Fetch for Tidal {
    type Loaded = Loaded;
    type Source = Opened;

    fn name(&self) -> &'static str {
        "tidal"
    }

    /// Asks for the chosen quality, or the best below it the account may play, and starts the
    /// download, waiting only for the preroll, and for that no longer than `PRIMING`.
    async fn load(&self, id: &str) -> Result<Loaded> {
        let playback = self.client.playback(id, self.quality).await?;
        log::info!(
            "tidal: {id} streams as {} {}",
            playback.quality,
            playback.manifest.codec()
        );
        let (stream, timeline) = tokio::time::timeout(PRIMING, self.arrive(id, playback.manifest))
            .await
            .context("the tidal track did not start arriving in time")??;
        Ok(Loaded {
            stream,
            timeline,
            duration: playback.duration,
            loudness: playback.loudness,
        })
    }

    fn length(&self, loaded: &Loaded) -> Option<Duration> {
        loaded.duration
    }

    fn loudness(&self, loaded: &Loaded) -> Option<Loudness> {
        loaded.loudness
    }

    /// A track that failed because Tidal refused the session is the listener having to sign
    /// in again, not the track being unplayable.
    fn gated(&self, error: &anyhow::Error) -> bool {
        crate::tidal::auth::refused(error)
    }

    /// Resolves once the last segment or the whole file is in, so the next track waits rather
    /// than taking bandwidth from this one.
    async fn downloaded(&self, loaded: &Loaded) {
        loaded.stream.finished().await;
    }

    /// A segmented stream has no index for symphonia to jump around in, so a seek opens a
    /// second decoder at the target. A file is simply opened there.
    fn reopen_to_seek(&self) -> bool {
        true
    }

    fn open(&self, id: &str, loaded: &Loaded, at: Duration) -> Option<Self::Source> {
        self.open_near(id, loaded, at).map(|(source, _)| source)
    }

    /// Builds a decoder placed at `at`, or at the start when `at` cannot be reached without
    /// decoding everything before it. Only the header is read here.
    ///
    /// A file is opened seekable and moved. A segmented track is opened without its length,
    /// which would have symphonia read every fragment to build a sample table, and a position
    /// splices the init segment onto the segment that covers it, trimming what comes before
    /// the position off the front. A position the download has not reached in time starts the
    /// track from the top instead, since winding there would hold the audio thread for as long
    /// as it takes to decode the way.
    fn open_near(&self, id: &str, loaded: &Loaded, at: Duration) -> Option<(Opened, Duration)> {
        let (source, start) = match &loaded.timeline {
            Some(timeline) => open_segments(id, &loaded.stream, timeline, at),
            None => open_file(id, &loaded.stream, at),
        }?;
        log::debug!(
            "playback: {id} decodes at {} Hz, {} channels",
            source.sample_rate(),
            source.channels()
        );
        Some((source, start))
    }
}

/// Opens a segmented track on the segment covering `at`, once the download has reached it.
fn open_segments(
    id: &str,
    stream: &Stream<Long>,
    timeline: &Mutex<Timeline>,
    at: Duration,
) -> Option<(Opened, Duration)> {
    let entry = match at.is_zero() {
        true => None,
        false => stream.awaiting(|_, _, _| lock(timeline).entry(at)),
    };
    let reader = match entry {
        Some(entry) => stream.spliced(entry.head, entry.graft),
        None => stream.reader(),
    };
    let decoder = decoder(
        id,
        rodio::Decoder::builder()
            .with_data(reader)
            .with_seekable(false),
    )?;
    let Some(entry) = entry else {
        if !at.is_zero() {
            log::warn!(
                "playback: the tidal track {id} has not arrived as far as {}s, starting it from the top",
                at.as_secs()
            );
        }
        return Some((Trimmed::new(decoder, Duration::ZERO, None), Duration::ZERO));
    };
    Some((Trimmed::new(decoder, entry.into, None), at))
}

/// Opens a track that is one file and moves it to `at`. A file that cannot be moved there is
/// opened again from the top, since a failed seek may leave its decoder anywhere.
fn open_file(id: &str, stream: &Stream<Long>, at: Duration) -> Option<(Opened, Duration)> {
    let mut decoder = file_decoder(id, stream)?;
    if at.is_zero() {
        return Some((Trimmed::new(decoder, Duration::ZERO, None), Duration::ZERO));
    }
    if let Err(error) = decoder.try_seek(at) {
        log::warn!(
            "playback: cannot start the tidal track {id} at {}s, starting it from the top: {error}",
            at.as_secs()
        );
        let decoder = file_decoder(id, stream)?;
        return Some((Trimmed::new(decoder, Duration::ZERO, None), Duration::ZERO));
    }
    Some((Trimmed::new(decoder, Duration::ZERO, None), at))
}

/// A seekable decoder over the whole of a track that is one file.
fn file_decoder(id: &str, stream: &Stream<Long>) -> Option<rodio::Decoder<Reader<Long>>> {
    let builder = rodio::Decoder::builder()
        .with_data(stream.reader())
        .with_seekable(true);
    let builder = match stream.total() {
        Some(total) => builder.with_byte_len(total),
        None => builder,
    };
    decoder(id, builder)
}

/// One request for a part, read whole. A part longer than `SEGMENT_CEILING` is refused before
/// it can fill memory, since nothing past here would stop it.
async fn whole(http: &reqwest::Client, url: &str) -> Result<Bytes, Failure> {
    let mut response = Failure::check(http.get(url).send().await?)?;
    let mut part = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if part.len() + chunk.len() > SEGMENT_CEILING {
            return Err(Failure::Final(anyhow!(
                "a segment runs past {SEGMENT_CEILING} bytes"
            )));
        }
        part.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(part))
}

/// The first byte and the full length a `Content-Range` header names. The length is `None`
/// when the server left it as `*`.
fn content_range(response: &reqwest::Response) -> Option<(u64, Option<u64>)> {
    let value = response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)?
        .to_str()
        .ok()?;
    let (range, total) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (start, _) = range.split_once('-')?;
    Some((start.trim().parse().ok()?, total.trim().parse().ok()))
}

/// The http client a track's bytes come through.
fn media_http() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(MEDIA_CONNECT)
        .read_timeout(MEDIA_STALL)
        .build()
        .context("cannot build the tidal media client")
}

/// Builds a decoder, or reports a track that will not decode.
fn decoder(
    id: &str,
    builder: rodio::decoder::DecoderBuilder<Reader<Long>>,
) -> Option<rodio::Decoder<Reader<Long>>> {
    match builder.build() {
        Ok(decoder) => Some(decoder),
        Err(error) => {
            log::warn!("playback: cannot decode the tidal track {id}: {error}");
            None
        }
    }
}
