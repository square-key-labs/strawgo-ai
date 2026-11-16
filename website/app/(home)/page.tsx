import Link from 'next/link';
import { ArrowRight, ArrowUpRight } from 'lucide-react';
import { ServerCodeBlock } from 'fumadocs-ui/components/codeblock.rsc';
import { Mermaid } from '@/components/mermaid';
import { repoUrl } from '@/lib/shared';

const install = 'go get github.com/square-key-labs/strawgo-ai@latest';

const pipelineCode = `stt := deepgram.NewSTTService(deepgram.STTConfig{
	APIKey:      os.Getenv("DEEPGRAM_API_KEY"),
	Model:       "nova-3",
	Language:    "en",
	SmartFormat: true,
})
llmSvc := openai.NewLLMService(openai.LLMConfig{
	APIKey:       os.Getenv("OPENAI_API_KEY"),
	Model:        "gpt-4.1-mini",
	SystemPrompt: systemPrompt,
})
tts := cartesia.NewTTSService(cartesia.TTSConfig{
	APIKey:   os.Getenv("CARTESIA_API_KEY"),
	VoiceID:  envOr("CARTESIA_VOICE_ID", "71a7ad14-091c-4e8e-a314-022ece01c121"),
	Model:    "sonic-2",
	Language: "en",
})

conversation := llm.NewContext()
pair := aggregators.NewContextAggregatorPair(conversation, aggregators.ContextAggregatorPairParams{})

procs = append(procs, stt, pair.User(), llmSvc, tts, s.Output(), pair.Assistant())
task := pipeline.NewPipelineTask(pipeline.NewPipeline(procs))`;

const callFlow = `flowchart LR
    caller((Caller)) <-->|audio over WebSocket| T[Transport]
    T --> In[Session input]
    In --> STT[STT]
    STT --> UA[User context aggregator]
    UA --> LLM[LLM]
    LLM --> TTS[TTS]
    TTS --> Out[Session output]
    Out --> AA[Assistant context aggregator]
    Out --> T`;

const areas = [
  {
    title: 'Getting started',
    href: '/getting-started/introduction',
    body: 'What a voice agent is, how a call flows through Strawgo, and a bot that answers you in five minutes.',
    links: [
      { label: 'Introduction', href: '/getting-started/introduction' },
      { label: 'Installation', href: '/getting-started/installation' },
      { label: 'Core concepts', href: '/getting-started/core-concepts' },
    ],
    variant: 'feature',
  },
  {
    title: 'Guides',
    href: '/guides/function-calling',
    body: 'Function calling, turn-taking, flows, telephony carriers, WebRTC, recording, observability and deployment.',
    links: [
      { label: 'Turn-taking', href: '/guides/turn-taking' },
      { label: 'Telephony providers', href: '/guides/telephony-providers' },
      { label: 'Deployment', href: '/guides/deployment' },
    ],
    variant: 'pattern',
  },
  {
    title: 'Reference',
    href: '/reference',
    body: 'Frames, processors, pipelines, transports, serializers and the rest of the public API.',
    links: [
      { label: 'Frames', href: '/reference/frames' },
      { label: 'Pipeline', href: '/reference/pipeline' },
    ],
    variant: 'plain',
  },
  {
    title: 'Providers',
    href: '/providers',
    body: 'Every LLM, STT, TTS and speech-to-speech vendor, with its package, constructor and test status.',
    links: [
      { label: 'Provider matrix', href: '/providers' },
      { label: 'Writing a custom provider', href: '/providers/custom' },
    ],
    variant: 'plain',
  },
  {
    title: 'Cookbook',
    href: '/cookbook',
    body: 'Complete programs for common jobs: support bots, bookings, outbound calls, failover and more.',
    links: [
      { label: 'Customer support bot', href: '/cookbook/customer-support-bot' },
      { label: 'LLM failover', href: '/cookbook/llm-failover' },
    ],
    variant: 'tint',
  },
] as const;

const paths = [
  { want: 'Use a carrier other than Twilio', href: '/guides/telephony-providers', page: 'Telephony providers' },
  { want: 'Build a browser voice bot', href: '/guides/webrtc', page: 'WebRTC' },
  { want: 'Let the bot call my APIs', href: '/guides/function-calling', page: 'Function calling' },
  { want: 'Script a structured conversation', href: '/guides/flows', page: 'Flows' },
  { want: 'Use OpenAI Realtime or Gemini Live', href: '/guides/speech-to-speech', page: 'Speech-to-speech' },
  { want: 'Tell a person from an answering machine', href: '/guides/voicemail-detection', page: 'Voicemail detection' },
  { want: 'Run it in production', href: '/guides/deployment', page: 'Deployment' },
  { want: 'Fix something that is not working', href: '/guides/troubleshooting', page: 'Troubleshooting' },
];

export default function HomePage() {
  return (
    <main className="sg-home">
      <section className="sg-hero">
        <div className="sg-hero-copy">
          <h1>Real-time voice agents, written in Go.</h1>
          <p className="sg-hero-sub">
            Strawgo streams a call through speech-to-text, an LLM and text-to-speech in one pipeline you control from Go.
          </p>
          <div className="sg-hero-install">
            <ServerCodeBlock code={install} lang="sh" codeblock={{ title: 'Terminal' }} />
          </div>
          <div className="sg-hero-actions">
            <Link href="/getting-started/quickstart" className="sg-btn sg-btn-primary">
              Quickstart
              <ArrowRight aria-hidden="true" />
            </Link>
            <Link href="/getting-started/first-phone-bot" className="sg-btn sg-btn-secondary">
              Build your first phone bot
            </Link>
          </div>
        </div>
        <div className="sg-hero-code">
          <ServerCodeBlock code={pipelineCode} lang="go" codeblock={{ title: 'main.go' }} />
          <p className="sg-caption">
            The heart of the Twilio bot from{' '}
            <Link href="/getting-started/first-phone-bot">Your first phone bot</Link>: Deepgram in, OpenAI in the
            middle, Cartesia out.
          </p>
        </div>
      </section>

      <section className="sg-flow" aria-labelledby="flow-title">
        <div className="sg-flow-text">
          <h2 id="flow-title">A call is a pipeline of frames</h2>
          <p>
            Audio, text and signals move through an ordered chain of processors as small units called frames. Each
            stage streams, so the bot starts speaking its first sentence while the LLM is still writing the second.
          </p>
          <Link href="/getting-started/core-concepts" className="sg-text-link">
            Learn the core concepts
            <ArrowRight aria-hidden="true" />
          </Link>
        </div>
        <Mermaid chart={callFlow} />
      </section>

      <section className="sg-areas" aria-labelledby="areas-title">
        <h2 id="areas-title">Find your way around</h2>
        <div className="sg-bento">
          {areas.map((area) => (
            <article key={area.title} className="sg-area" data-variant={area.variant}>
              <h3>
                <Link href={area.href} className="sg-area-link">
                  {area.title}
                  <ArrowRight aria-hidden="true" />
                </Link>
              </h3>
              <p>{area.body}</p>
              <ul>
                {area.links.map((link) => (
                  <li key={link.href}>
                    <Link href={link.href}>{link.label}</Link>
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

      <section className="sg-paths" aria-labelledby="paths-title">
        <h2 id="paths-title">I want to...</h2>
        <ul className="sg-path-list">
          {paths.map((path) => (
            <li key={path.href}>
              <Link href={path.href}>
                <span className="sg-path-want">{path.want}</span>
                <span className="sg-path-page">
                  {path.page}
                  <ArrowRight aria-hidden="true" />
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </section>

      <section className="sg-status" aria-labelledby="status-title">
        <h2 id="status-title">Know what has been tested</h2>
        <p>
          Provider integrations are tested against mock servers at best, some not at all, and none against a live
          vendor API. Check the status of each one before you depend on it.
        </p>
        <Link href="/providers" className="sg-text-link">
          Open the provider matrix
          <ArrowRight aria-hidden="true" />
        </Link>
      </section>

      <footer className="sg-footer">
        <span className="sg-footer-note">Strawgo is MIT licensed. These pages are built from the docs folder of the repository.</span>
        <nav aria-label="Project links" className="sg-footer-links">
          <a href={repoUrl}>
            GitHub
            <ArrowUpRight aria-hidden="true" />
          </a>
          <a href={`${repoUrl}/blob/main/CHANGELOG.md`}>
            Changelog
            <ArrowUpRight aria-hidden="true" />
          </a>
          <Link href="/overview">Documentation map</Link>
        </nav>
      </footer>
    </main>
  );
}
