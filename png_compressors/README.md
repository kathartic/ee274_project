# PNG compressors

All the compressors in this file are variations on PNG -- see RFC 2083 for more
details.

## Implementation

To function with the script(s) in `analysis/` each compressor inherits from the
parent `CoreEncoder` class. `CoreEncoder` implements a method `encode_image()`
that:

1.  Takes a PIL.Image
2.  Breaks it up by color channel
3.  Feeds each channel to the `encode_block()` method implemented by the child
    class of `CoreEncoder()`.
4.  Concatenates each encoded color channel

As such, each child class of `CoreEncoder` is only responsible for implementing
`encode_block()`. The implicit assumption is that each channel is independent of
the other, which of course is not universally so.

## Suggestions on `encode_block()`

When adding a compressor to this folder, it's suggested that you take advantage
of the existing PNG filters. While it's not necessary to encode according to the
established heuristic (see `png_tools/png_filters.py`), it may make the final
encoded size smaller.


