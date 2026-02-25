<?php

declare(strict_types=1);

namespace PhpMlKit\NDArray\Traits;

use PhpMlKit\NDArray\FFI\Lib;

trait CanBePrinted
{
    /**
     * @var array{threshold: int, edgeitems: int, precision: int}
     */
    private static array $printOptions = [
        'threshold' => 1000,
        'edgeitems' => 3,
        'precision' => 8,
    ];

    /**
     * Set global print options for string representation.
     *
     * These options will be used when casting to string or echoing the array.
     *
     * @param int $threshold Maximum number of elements before truncation is applied.
     *                      Arrays with more elements than this will be truncated.
     *                      Default: 1000
     * @param int $edgeitems Number of items to show at each edge when truncating.
     *                        Default: 3
     * @param int $precision Number of decimal places for floating-point numbers.
     *                        Default: 8
     */
    public static function setPrintOptions(
        int $threshold = 1000,
        int $edgeitems = 3,
        int $precision = 8,
    ): void {
        self::$printOptions = [
            'threshold' => $threshold,
            'edgeitems' => $edgeitems,
            'precision' => $precision,
        ];
    }

    /**
     * Get the current print options.
     *
     * @return array{threshold: int, edgeitems: int, precision: int}
     */
    public static function getPrintOptions(): array
    {
        return self::$printOptions;
    }

    /**
     * Reset print options to their default values.
     */
    public static function resetPrintOptions(): void
    {
        self::$printOptions = [
            'threshold' => 1000,
            'edgeitems' => 3,
            'precision' => 8,
        ];
    }

    /**
     * Get string representation of the array (magic method).
     *
     * Uses the global print options configured via setPrintOptions().
     */
    public function __toString(): string
    {
        $options = self::$printOptions;

        $ffi = Lib::get();
        $buffer = $ffi->new('char[8192]');

        $meta = $this->viewMetadata()->toCData();
        $len = $ffi->ndarray_to_string(
            $this->handle,
            Lib::addr($meta),
            $buffer,
            8192,
            $options['threshold'],
            $options['edgeitems'],
            $options['precision']
        );

        if ($len === 0) {
            return '[Error: Failed to format array]';
        }

        if ($len > 8192) {
            $newBuffer = $ffi->new("char[{$len}]");
            $len = $ffi->ndarray_to_string(
                $this->handle,
                Lib::addr($meta),
                $newBuffer,
                $len,
                $options['threshold'],
                $options['edgeitems'],
                $options['precision']
            );

            $formatted = \FFI::string($newBuffer, $len);
        } else {
            $formatted = \FFI::string($buffer, $len);
        }

        $header = $this->formatHeader();

        return $header . $formatted;
    }

    /**
     * Format the header line showing array shape.
     */
    private function formatHeader(): string
    {
        $shape = $this->shape();

        if (empty($shape)) {
            return "array(0)\n";
        }

        return 'array(' . implode(', ', $shape) . ")\n";
    }
}
