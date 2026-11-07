# tessella-cli

> Command-line interface to interact with the network.

## Responsibility

Allows users to send prompts to the distributed network and receive token-by-token responses (streaming). Formats prompts according to model templates (ChatML or FIM) and resolves corresponding EOS tokens. Depends on `tessella-core` and `reqwest`.

## Modules

| File | Content |
|---------|-----------|
| `src/main.rs` | CLI entrypoint and HTTP streaming logic. |

## Usage

```bash
cargo run --bin tessella-cli -- "Write a loop in Rust"
```
