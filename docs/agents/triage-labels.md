# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the actual `Status:` strings used in this repo's issue files under `.scratch/`.

| Role              | `Status:` string  | Meaning                                  |
| ----------------- | ----------------- | ---------------------------------------- |
| `needs-triage`    | `needs-triage`    | Maintainer needs to evaluate this issue  |
| `needs-info`      | `needs-info`      | Waiting on reporter for more information |
| `ready-for-agent` | `ready-for-agent` | Fully specified, ready for an AFK agent  |
| `ready-for-human` | `ready-for-human` | Requires human implementation            |
| `wontfix`         | `wontfix`         | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding string from this table on the `Status:` line of the issue file.

Edit the right-hand column to match whatever vocabulary you actually use.
