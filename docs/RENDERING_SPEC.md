# Offline Rendering Specification

**Status:** Not started  
**Goal:** Turn a composition (list of MusicalEvents + context) into deterministic stereo audio and stems.

## Required capabilities

- Load composition / event stream
- Generate control frames via adapter
- Schedule multiple voices (once polyphony exists)
- Render at selectable sample rate
- Produce stereo audio
- Support variable duration and tails
- Export individual instrument stems
- Deterministic renders from the same inputs + model versions
- Report failures clearly

## Recommended initial formats

| Format             | Purpose                     |
|--------------------|-----------------------------|
| WAV 24-bit         | Production master           |
| WAV 32-bit float   | Mixing / further processing |
| MIDI               | Musical interchange         |
| JSON sidecar       | Events + parameter frames   |
| Stems              | Per-instrument groups       |

## Determinism requirements

- Same composition + same controller checkpoint + same ranges → bit-identical audio (or documented floating-point tolerance)
- Contract versions and model hashes recorded in the JSON sidecar
