$ErrorActionPreference = "Stop"

Describe "doctor demo port preflight" {
    It "fails when a required demo port is already occupied" {
        $listeners = @()
        foreach ($port in @(8000, 5173)) {
            try {
                $listener = New-Object System.Net.Sockets.TcpListener(
                    [System.Net.IPAddress]::Loopback,
                    $port
                )
                $listener.Start()
                $listeners += $listener
            } catch [System.Net.Sockets.SocketException] {
                # An existing demo listener exercises the same public behavior.
            }
        }

        try {
            $doctor = Join-Path $PSScriptRoot "..\doctor.ps1"
            $output = & powershell.exe `
                -NoProfile `
                -ExecutionPolicy Bypass `
                -File $doctor `
                -ProviderMode mock 2>&1
            $exitCode = $LASTEXITCODE
        } finally {
            foreach ($listener in $listeners) {
                $listener.Stop()
            }
        }

        $exitCode | Should Not Be 0
        ($output -join "`n") | Should Match "\[FAIL\] Port (8000|5173)"
    }
}
