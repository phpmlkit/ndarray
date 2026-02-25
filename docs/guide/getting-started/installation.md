# Installation

## Requirements

Before installing NDArray, ensure your system meets these requirements:

- **PHP**: Version 8.1 or higher
- **FFI Extension**: Must be enabled in your PHP installation
- **Composer**: For dependency management

## Check FFI Extension

```bash
php -m | grep FFI
```

If you don't see "FFI" in the output, you need to enable it.

### Enabling FFI Extension

#### On Linux/macOS

Edit your `php.ini` file (location varies by installation):

```bash
# Find your php.ini
php --ini

# Edit the file and add:
extension=ffi
```

#### On Windows (XAMPP/WAMP)

1. Open `php.ini` in your PHP installation directory
2. Find `;extension=ffi`
3. Remove the semicolon: `extension=ffi`
4. Restart Apache

## Install via Composer

```bash
composer require phpmlkit/ndarray
```

## Installation Verification

Create a test file to verify everything is working:

```php
<?php

require_once 'vendor/autoload.php';

use PhpMlKit\NDArray\NDArray;

// Create a simple array
$vector = NDArray::array([1, 2, 3, 4, 5]);
echo "Vector: $vector\n";
echo "Sum: {$vector->sum()}\n\n";

// Create a 2D array
$matrix = NDArray::zeros([3, 3]);
echo "Zero matrix: $matrix\n\n";

echo "✓ NDArray is working correctly!\n";
```

Run it:

```bash
php test.php
```

Expected output:

```
Vector: array(5)
[1 2 3 4 5]
Sum: 15

Zero matrix: array(3, 3)
[
 [0 0 0]
 [0 0 0]
 [0 0 0]
]

✓ NDArray is working correctly!
```

## Troubleshooting

### "FFI extension not loaded"

Make sure FFI is enabled in the PHP.ini file that the CLI uses:

```bash
php --ini
```

Edit that specific file and ensure `extension=ffi` is present (without the semicolon).

### "Platform package not found"

If you see errors about platform-specific binaries not being found:

1. Check your platform is supported (Linux x86_64/ARM64, macOS x86_64/ARM64, Windows x86_64)
2. Try clearing Composer cache: `composer clear-cache`
3. Reinstall: `composer require phpmlkit/ndarray`

### Permission Errors

On Linux/macOS, you might need to set execute permissions:

```bash
chmod +x vendor/phpmlkit/ndarray/bin/*
```

### Memory Limit Issues

For large arrays, increase PHP's memory limit:

```bash
php -d memory_limit=512M your-script.php
```

Or in your script:

```php
ini_set('memory_limit', '512M');
```

## Next Steps

Now that NDArray is installed:

- **[Quick Start](/guide/getting-started/quick-start)** - Create your first arrays
- **[NumPy Migration](/guide/getting-started/numpy-migration)** - If coming from Python
