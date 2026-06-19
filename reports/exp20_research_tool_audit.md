# Exp20 Research Tool Audit

## Tools

| Tool | Local path | Commit | Use |
| --- | --- | --- | --- |
| autoresearch | `/home/hj/third_party/research_tools/autoresearch` | `228791fb499afffb54b46200aca536f79142f117` | design reference only |
| AI-Scientist-v2 | `/home/hj/third_party/research_tools/AI-Scientist-v2` | `96bd51617cfdbb494a9fc283af00fe090edfae48` | design reference only |

## Borrowed Ideas

- fixed wall-clock trial budget;
- immutable evaluator;
- explicit keep/discard/crash records;
- parent-child experiment nodes;
- best-first expansion;
- parallel workers;
- debug depth limit;
- append-only logs.

## Not Borrowed

- no self-modifying Python during sweep;
- no `git reset`;
- no LLM-written code execution;
- no autonomous package install;
- no AI-Scientist manuscript/writeup pipeline;
- no copied source code into the project.

## Risk / License Notes

`autoresearch` is MIT. AI-Scientist-v2 uses a responsible-use source code
license and warns that it executes LLM-written code. Because this project only
uses high-level design ideas and does not run or copy AI-Scientist-v2 code,
license/disclosure risk for Exp20 implementation is low. If any future paper
uses AI-Scientist-v2 directly, disclosure would be required.
