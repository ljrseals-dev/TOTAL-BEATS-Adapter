# Album Production Specification

**Status:** Target architecture only  
**Goal:** Complete, reproducible album deliverables from TOTAL BEATS compositions.

## Pipeline stages

1. Album identity (genre, concept, title, sonic direction, artwork)
2. Composition (TOTAL BEATS)
3. Instrumentation selection
4. Performance generation (expressive events)
5. Synthesis (adapter → Sogitech)
6. Arrangement and mixing
7. Mastering
8. Packaging and quality control

## Required deliverables

| Deliverable            | Contents                                      |
|------------------------|-----------------------------------------------|
| Album masters          | One final WAV per track                       |
| Instrumental masters   | Instrumental versions                         |
| Stems                  | Separated instrument groups                   |
| MIDI                   | Reproducible performances                     |
| Project manifests      | Composition + synthesis settings + versions   |
| Track metadata         | Title, duration, credits, sequence            |
| Artwork                | Cover and supporting art                      |
| Quality report         | Technical verification of every track         |
| Release package        | Organised, ready-to-distribute folder         |

## Quality gates (per track)

- No missing audio / unexpected silence
- No clipping / true-peak compliance
- No abrupt endings
- Balanced instrument levels
- Spectral energy within expected bounds
- Stereo phase check
- Correct sample rate and duration
- Metadata consistency

Album-level: consistent loudness, tonal identity, sequencing.
