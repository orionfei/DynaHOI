## Purpose

This project prioritizes correctness, clarity, and faithful algorithmic implementation over robustness hacks or defensive programming.

Code should reflect the intended algorithm directly, not a patched or over-engineered approximation.

---

## Core Principles

1. Prefer minimal, direct, and readable implementations.
2. Implement the true algorithm, not a workaround.
3. Expose errors instead of hiding them.
4. Fix root causes, not symptoms.

---

## Strict Rules (Must Follow)

### 1. No fallback or silent recovery

Do NOT:
- add fallback logic
- silently replace invalid inputs with defaults
- retry operations without explicit reason
- "try best effort" behavior

If something is invalid → **fail explicitly**

---

### 2. No hacks or patch-based fixes

Do NOT:
- add special-case branches just to pass current inputs/tests
- introduce ad hoc heuristics
- patch outputs after computation to "look correct"
- clamp / normalize / adjust values unless required by the algorithm

---

### 3. No silent error masking

Do NOT:
- use broad try/except to suppress errors
- return placeholder values (0, None, empty) when logic fails
- ignore shape mismatches, NaNs, or invalid states

Instead:
- raise clear exceptions
- use assertions for invariants

---

### 4. Fail fast

When assumptions are violated:
- stop execution immediately
- raise explicit, informative errors

Prefer:
- `assert` for invariants
- `ValueError` / `RuntimeError` for invalid inputs

---

### 5. No fake robustness

Do NOT add code just to make things "more stable" if it is not part of the real algorithm:
- no smoothing
- no clamping
- no auto-correction
- no hidden normalization

If the algorithm is unstable → expose it.

---

### 6. No over-engineering

Do NOT:
- add unnecessary abstraction layers
- introduce generic frameworks prematurely
- split simple logic into many small wrappers

Prefer:
- simple functions
- direct data flow
- explicit logic

---

### 7. Unsupported cases must be explicit

If something is not supported:
- state it clearly
- raise an error

Do NOT:
- guess behavior
- approximate silently
- degrade functionality

---

## Decision Checklist

Before adding any extra logic, ask:

1. Is this required by the actual algorithm?
2. Would this exist in a clean reference implementation?
3. Does this expose bugs, or hide them?

If it hides bugs → **DO NOT ADD IT**

---

## Preferred Style

Good code in this project:

- minimal
- explicit
- deterministic
- easy to reason about
- fails loudly when wrong

Bad code in this project:

- defensive
- overly tolerant
- full of fallbacks
- silently correcting errors
- hard to trace

---

## Error Handling Policy

Allowed:
- explicit validation
- assertions
- precise exceptions

Not allowed:
- silent fixes
- implicit conversions
- hidden retries
- swallowing exceptions

---

## Notes for Code Generation Agents (Codex)

When generating or modifying code:

- Do not add fallback paths
- Do not add "just in case" handling
- Do not attempt to make code more robust than specified
- Do not hide incorrect behavior

If uncertain:
→ prefer a clean, strict implementation that may fail

Correctness > robustness  
Clarity > coverage  
Failure visibility > silent success