# set_discriminant

Positive test: verifies that the mir-importer lowers MIR
`StatementKind::SetDiscriminant` to a device-side enum tag write, rather than
silently dropping the discriminant update.

## What this tests

The example calls a `custom_mir` helper from a `#[kernel]` to emit
`StatementKind::SetDiscriminant` directly. Each thread starts with
`DeviceState::Full(idx)`, sets the discriminant to `Empty`, and writes `1` to
output if the new variant is observed.

Before the lowering was implemented, this produced:

```
Unsupported construct: SetDiscriminant statements are not yet supported on the device;
until they are lowered, enum discriminant writes would be silently dropped
```

## Usage

```bash
cargo oxide run set_discriminant
```

## Expected output

The build succeeds and the host verifies that every thread observed the
discriminant write:

```
PASS: all 64 threads observed the SetDiscriminant write.
```
