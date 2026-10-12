# AWS IoT Core credentials (not committed)

Create a Thing in AWS IoT Core, attach a policy that allows `iot:Connect` with
client ID = thing name and `iot:Publish` / `iot:Subscribe` / `iot:Receive` on
`edgewake/<thing>/*`, then download its certificate and place three files here:

| File | What it is |
|------|------------|
| `AmazonRootCA1.pem` | Amazon Root CA 1 (verifies the broker) |
| `device.pem.crt`    | Device certificate (mutual TLS client cert) |
| `private.pem.key`   | Device private key |

These names are git-ignored. When they are missing the build prints a warning
and the firmware runs offline.
