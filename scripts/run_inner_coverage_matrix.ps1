param(
    [string]$Build = "runtime/build",
    [string]$Config = "RelWithDebInfo",
    [string]$Glslang = "",
    [string]$SpirvAs = "",
    [string]$SpirvVal = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$matrixBuild = if ([IO.Path]::IsPathRooted($Build)) { $Build } else { Join-Path $root $Build }
$configureArgs = @("-S", (Join-Path $root "runtime"), "-B", $matrixBuild,
    "-DLEONIDA_TEST_INNER_COVERAGE_MATRIX=ON", "-DLEONIDA_VULKAN_HEADERS_FETCH=ON")
if ($Glslang) { $configureArgs += "-DLEONIDA_INNER_GLSLANG=$Glslang" }
if ($SpirvAs) { $configureArgs += "-DLEONIDA_INNER_SPIRV_AS=$SpirvAs" }
if ($SpirvVal) { $configureArgs += "-DLEONIDA_INNER_SPIRV_VAL=$SpirvVal" }
& cmake @configureArgs
if ($LASTEXITCODE -ne 0) { throw "InnerCoverage matrix configure failed ($LASTEXITCODE)." }
& cmake --build $matrixBuild --config $Config --target vk_inner_coverage
if ($LASTEXITCODE -ne 0) { throw "InnerCoverage harness build failed ($LASTEXITCODE)." }
& ctest --test-dir $matrixBuild -C $Config -R '^inner_coverage_matrix$' --no-tests=error -V
if ($LASTEXITCODE -ne 0) { throw "InnerCoverage matrix failed ($LASTEXITCODE)." }
# CTest reports hardware omissions as Skipped. They are not positive GPU validation.
