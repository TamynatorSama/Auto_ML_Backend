from automl_sandbox.output import OutputBuffer


def test_reads_from_offset():
    buf = OutputBuffer(100)
    buf.write(b"hello ")
    buf.write(b"world")
    assert buf.read(0) == ("hello world", 11, False)
    assert buf.read(6) == ("world", 11, False)
    assert buf.read(11) == ("", 11, False)


def test_keeps_newest_bytes_and_reports_skipped():
    buf = OutputBuffer(5)
    buf.write(b"abcdefgh")
    assert buf.read(0) == ("defgh", 8, True)
    assert buf.read(3) == ("defgh", 8, False)
    assert buf.read(6) == ("gh", 8, False)


def test_offset_past_end_is_clamped():
    buf = OutputBuffer(10)
    buf.write(b"abc")
    assert buf.read(50) == ("", 3, False)
    assert buf.read(-4) == ("abc", 3, False)


def test_bad_utf8_is_replaced():
    buf = OutputBuffer(10)
    buf.write(b"a\xffb")
    assert buf.read(0)[0] == "a�b"
