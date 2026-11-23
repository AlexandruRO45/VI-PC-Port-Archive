"""gamedata_xml.scan_xml (callback pass) must give what the iterparse scan it replaced gave.

    python -I tests/test_gamedata_xml_fast_scan.py

``reference_scan`` below is that scan, kept inline: the checks compare the tool with it on the same bytes,
so nothing here is pinned to a platform, a hash or a byte count. Documents are hand-written or drawn from a
seeded generator. No game data is used.
"""
import io
import os
import random
import re
import sys
import tracemalloc
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from xml.parsers.expat import errors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gamedata_xml as g  # noqa: E402

MACRO_REF = re.compile(r"\$\(([A-Za-z_][\w.-]*)\)")
PLATFORM_REF = re.compile(r"platform:/([A-Za-z0-9_.-]+)?")
XI_INCLUDE = "{http://www.w3.org/2001/XInclude}include"
COMMENT_MARKUP = re.compile(
    rb"(?P<comment><!--).*?(?:-->|\Z)|<!\[CDATA\[.*?(?:\]\]>|\Z)|<\?.*?(?:\?>|\Z)", re.S)


def ref_text(text, report):
    if not text:
        return
    if "$(" in text:
        for match in MACRO_REF.finditer(text):
            report["refs"]["$(" + match.group(1) + ")"] += 1
    if "platform:/" in text:
        for match in PLATFORM_REF.finditer(text):
            report["platform"]["levels" if (match.group(1) or "").lower() == "levels" else "other"] += 1
            report["platform_total"] += 1


def reference_scan(data):
    """The element-tree/iterparse scan, as it was before the callback pass."""
    report = {
        "root": "", "well_formed": False, "mismatched_tag": False, "error": "", "element_count": 0,
        "max_depth": 0, "attributes": [], "comments": 0, "elements": Counter(), "attrs": Counter(),
        "refs": Counter(), "platform": Counter(), "platform_total": 0, "filename": 0, "parent_path": 0,
        "parent_file": 0, "xi_include": 0, "type_attrs": 0, "type_values": set(), "content_values": set(),
    }
    depth = 0
    ended = None
    try:
        for event, elem in ET.iterparse(io.BytesIO(data), events=("start", "end", "comment")):
            if event == "comment":
                report["comments"] += 1
                continue
            if ended is not None:
                ref_text(ended.tail, report)
                ended.clear()
                ended = None
            if event == "start":
                depth += 1
                report["max_depth"] = max(report["max_depth"], depth)
                if not report["root"]:
                    report["root"] = elem.tag
                report["element_count"] += 1
                report["elements"][elem.tag] += 1
                for name, value in elem.attrib.items():
                    report["attrs"][name] += 1
                    if name == "type":
                        report["type_attrs"] += 1
                        report["type_values"].add(value)
                    elif name == "content":
                        report["content_values"].add(value)
                    elif name == "parentPath":
                        report["parent_path"] += 1
                    elif name == "parentFile":
                        report["parent_file"] += 1
                    ref_text(value, report)
                if elem.tag == "filename":
                    report["filename"] += 1
                if elem.tag == XI_INCLUDE or elem.tag == "xi:include":
                    report["xi_include"] += 1
            else:
                ref_text(elem.text, report)
                depth -= 1
                ended = elem
        report["well_formed"] = True
    except (ET.ParseError, LookupError, ValueError) as exc:
        report["error"] = str(exc)
        report["mismatched_tag"] = (isinstance(exc, ET.ParseError)
                                    and exc.code == errors.codes[errors.XML_ERROR_TAG_MISMATCH])
        report["comments"] = sum(m.lastgroup == "comment" for m in COMMENT_MARKUP.finditer(data))
    if ended is not None:
        ref_text(ended.tail, report)
    report["attributes"] = sorted(report["attrs"])
    return report


XI = b' xmlns:xi="http://www.w3.org/2001/XInclude"'
DOCS = {
    "plain": b'<?xml version="1.0" encoding="UTF-8"?>\n<Root><Item type="a" content="b" id="1"/>\n</Root>',
    "bom": b'\xef\xbb\xbf<?xml version="1.0" encoding="utf-8"?><Root><a>x</a></Root>',
    "no declaration": b"<Root><a type='q'>t</a><a type='q'>u</a></Root>",
    "macros and platform": (b'<R a="$(m)/x" b="platform:/levels/z" c="platform:/Levels"><filename>$(a)$(b)</filename>'
                            b'<f>platform:/data/q platform:/ $(</f><g>$(bad name)</g></R>'),
    "references in a tail": b"<R><a/>tail $(t) platform:/levels/x<b/>$(u)</R>",
    "token split by a comment": b"<R>plat<!--x-->form:/levels $(a<!--y-->b)</R>",
    "token split by cdata": b"<R>plat<![CDATA[form:/]]>levels $<![CDATA[(a)]]></R>",
    "token split by a pi": b"<R>$<?pi x?>(a) platform<?q?>:/other</R>",
    "character reference": b'<R a="&#36;(c)">&#36;(d) platform&#58;/levels &amp;$(e)</R>',
    "doctype entity": b'<!DOCTYPE R [<!ENTITY e "$(ent) platform:/levels">]><R>&e;<a b="&e;"/></R>',
    "doctype only": b"<!DOCTYPE R><R><a/></R>",
    "namespaces": (b'<r xmlns="urn:x" xmlns:a="urn:a"><a:i a:k="1"><c>t</c></a:i><i k="$(n)"/></r>'),
    "xinclude": b'<R' + XI + b'><xi:include href="a"/><xi:include href="b"/></R>',
    "include-like": (b'<R><fileList><filename>a</filename><filename>b</filename></fileList>'
                     b'<includedSets/><includedChangesets/><changesetIncludeFlags/><loadPath>p</loadPath>'
                     b'<Item parentPath="x" parentFile="y" type="t1"/><Item type="t2" content="c1"/>'
                     b'<Item type="t1" content="c1"/></R>'),
    "comments": b"<!-- a --><R><!-- b --><x/><!-- c --></R><!-- d -->",
    "non-ascii utf-8": "<R a=\"é$(u)\">é platform:/levels ü</R>".encode("utf-8"),
    "latin-1": '<?xml version="1.0" encoding="ISO-8859-1"?><R a="é">é $(l)</R>'.encode("latin-1"),
    "utf-16 bom": '<?xml version="1.0"?><R a="$(w)">platform:/levels</R>'.encode("utf-16"),
    "utf-16 no bom": "<R a='$(w)'>platform:/levels</R>".encode("utf-16-le"),
    "ebcdic": '<?xml version="1.0" encoding="cp037"?><R a="$(e)">platform:/levels</R>'.encode("cp037"),
    "declared single byte": b'<?xml version="1.0" encoding="windows-1250"?><R a="$(e)">platform:/levels</R>',
    "declared unknown": b'<?xml version="1.0" encoding="no-such-codec"?><R a="$(e)">platform:/levels</R>',
    "declared utf-16 in ascii": b'<?xml version="1.0" encoding="UTF-16"?><R>$(x)</R>',
    "deep": b"<a>" * 300 + b"$(deep)" + b"</a>" * 300,
}
BROKEN = {
    "empty": b"",
    "text only": b"plain text, not a document",
    "mismatched": b'<?xml version="1.0"?><Root><Item a="$(m)"></Root>',
    "tail only": b'<Item type="x" a="$(m)">platform:/levels/t</Item>\n</TailRoot>',
    "truncated": b"<R><a type='x'>$(m)</a><b>platform:/levels/x</b><c",
    "truncated after comment": b"<R><!-- a --><x/><!-- b --><y",
    "unclosed comment": b"<R><x/><!-- never closed $(a)",
    "two roots": b"<a/><b/>",
    "undefined entity": b"<R>&nope;$(u)</R>",
    "duplicate attribute": b'<R a="1" a="2">$(d)</R>',
    "bad bytes": b"<R>\xff\xfe$(b)</R>",
}


def sample(rnd):
    tags = ["Item", "filename", "fileList", "xi:include", "a:b", "x", "root", "loadPath"]
    texts = ["t", "$(common)/x", "platform:/levels/a", "platform:/data/b", "$(", "platform:/", "&#36;(c)",
             "$(a)$(b)", "&amp;$(m)", "  \n ", "é$(u)", "<![CDATA[$(cd)]]>", "<!--c-->", "<?p $(q)?>",
             "$(s<!--c-->p)", "plat<!--x-->form:/levels", "platform<![CDATA[:/]]>levels"]
    attrs = ["type", "content", "parentPath", "parentFile", "name", "a:k"]
    values = ["file", "$(m)/y", "platform:/levels/z", "platform:/other", "&#36;(q)", "v&amp;w", "", "é"]

    def element(level):
        tag = rnd.choice(tags)
        given = {rnd.choice(attrs): rnd.choice(values) for _ in range(rnd.randrange(4))}
        decl = "".join(' %s="%s"' % item for item in given.items())
        decl += ' xmlns:a="urn:a" xmlns:xi="http://www.w3.org/2001/XInclude"' if rnd.random() < 0.5 else ""
        if ("a:" in tag or "xi:" in tag or "a:k" in given) and "xmlns" not in decl:
            decl += ' xmlns:a="urn:a" xmlns:xi="http://www.w3.org/2001/XInclude"'
        if rnd.random() < 0.2:
            return "<%s%s/>" % (tag, decl)
        parts = [element(level + 1) if level < 4 and rnd.random() < 0.5 else rnd.choice(texts)
                 for _ in range(rnd.randrange(5))]
        return "<%s%s>%s</%s>" % (tag, decl, "".join(parts), tag)

    head = rnd.choice(["", '<?xml version="1.0"?>', '<?xml version="1.0" encoding="UTF-8"?>',
                       "<?xml version='1.0' encoding='utf-8'?>", "<!-- lead -->",
                       '<!DOCTYPE x [<!ENTITY e "$(ent)">]>'])
    data = (head + element(0) + rnd.choice(["", "\n", "<!-- tail $(t) -->"])).encode("utf-8")
    kind = rnd.random()
    if kind < 0.06:
        data = b"\xef\xbb\xbf" + data
    elif kind < 0.09:
        data = data.decode("utf-8").encode("utf-16")
    elif kind < 0.11:
        data = data.decode("utf-8").encode("cp037")
    mutation = rnd.random()
    if mutation < 0.15 and len(data) > 4:
        data = data[:rnd.randrange(1, len(data))]
    elif mutation < 0.25 and len(data) > 4:
        at = rnd.randrange(len(data))
        data = data[:at] + bytes([rnd.randrange(256)]) + data[at + 1:]
    elif mutation < 0.3:
        data = data[rnd.randrange(len(data)):]
    return data


def big_document(items, broken_at=None):
    """A document of many 40-odd byte items that crosses the 16 KiB feed boundary of iterparse."""
    body = b"".join(b'<Item type="t%d" a="$(m%d)">platform:/levels/%d</Item>\n' % (i % 9, i, i)
                    for i in range(items))
    data = b"<R>\n" + body + b"</R>"
    return data if broken_at is None else data[:broken_at] + b"</Z>"


class FastScanParity(unittest.TestCase):
    def check(self, data, label=""):
        self.assertEqual(g.scan_xml(data), reference_scan(data), label or data[:60])

    def test_hand_written_documents(self):
        for label, data in DOCS.items():
            with self.subTest(label):
                self.check(data, label)

    def test_malformed_and_truncated_documents(self):
        for label, data in BROKEN.items():
            with self.subTest(label):
                self.check(data, label)
                self.assertFalse(g.scan_xml(data)["well_formed"])

    def test_every_prefix_of_a_document(self):
        data = DOCS["macros and platform"] + b"<!-- end -->"
        for end in range(len(data) + 1):
            with self.subTest(end=end):
                self.check(data[:end])

    def test_errors_past_the_first_feed_chunk_keep_the_facts_before_them(self):
        # iterparse reports what it read before the chunk that failed, so the counts depend on where the
        # error is: the fallback must reproduce that for an error in the 2nd, 3rd and 5th chunk.
        for at in (20_000, 40_000, 70_000):
            data = big_document(2500, broken_at=at)
            with self.subTest(at=at):
                scan = g.scan_xml(data)
                self.assertFalse(scan["well_formed"])
                self.assertEqual(scan, reference_scan(data))
                self.assertGreater(sum(scan["refs"].values()), 0)

    def test_well_formed_document_across_feed_chunks(self):
        data = big_document(2500)
        scan = g.scan_xml(data)
        self.assertTrue(scan["well_formed"])
        self.assertEqual(scan, reference_scan(data))
        self.assertGreater(len(data), 3 * g.FEED_CHUNK)

    def test_every_feed_chunk_size_gives_the_same_scan(self):
        data = big_document(300)
        want = reference_scan(data)
        saved = g.FEED_CHUNK
        try:
            for size in (1, 7, 64, 1000, 1 << 30):
                g.FEED_CHUNK = size
                with self.subTest(size=size):
                    self.assertEqual(g.scan_xml(data), want)
        finally:
            g.FEED_CHUNK = saved

    def test_seeded_random_documents(self):
        rnd = random.Random(20261123)
        well_formed = 0
        for index in range(1500):
            data = sample(rnd)
            with self.subTest(index=index):
                scan = g.scan_xml(data)
                self.assertEqual(scan, reference_scan(data), data[:100])
                well_formed += scan["well_formed"]
        self.assertGreater(well_formed, 300)         # the draw covers both kinds, not only broken ones

    def test_file_rows_match(self):
        rnd = random.Random(7)
        docs = [dict(data=data, from_carve=False, name="d%d.xml" % i, root_hint="")
                for i, data in enumerate(list(DOCS.values()) + list(BROKEN.values())
                                          + [sample(rnd) for _ in range(200)])]
        for doc in docs:
            doc["root_hint"] = g.gi.xml_root(doc["data"])
            with self.subTest(doc["name"]):
                self.assertEqual(g.file_row(doc, g.scan_xml(doc["data"])),
                                 g.file_row(doc, reference_scan(doc["data"])))

    def test_scan_xml_events_is_the_reference(self):
        for data in list(DOCS.values()) + list(BROKEN.values()):
            self.assertEqual(g.scan_xml_events(data), reference_scan(data))


class RefsPossible(unittest.TestCase):
    def test_plain_documents_have_no_references(self):
        for data in (b"<R/>", b'<?xml version="1.0"?><R a="b">t</R>',
                     b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?><R>\xc3\xa9</R>',
                     b"<?xml version='1.0' encoding='iso-8859-1'?><R/>",
                     b'<?xml version="1.0" encoding="US-ASCII"?><R/>'):
            with self.subTest(data=data):
                self.assertFalse(g.refs_possible(data))

    def test_anything_that_can_spell_a_reference_is_possible(self):
        for data in (b"<R>$(a)</R>", b"<R a='platform:/x'/>", b"<R>&#36;(a)</R>", b"<R>&amp;</R>",
                     b"<R>a<!--c-->b</R>", b"<R><![CDATA[x]]></R>", b"<R><?pi x?></R>",
                     b'<?xml version="1.0"?><R><?pi x?></R>',
                     "<R/>".encode("utf-16"), "<R/>".encode("utf-16-le"), "<R/>".encode("utf-16-be"),
                     b"text", b"", b"\x4c\x6f\xa7\x94\x93",
                     b'<?xml version="1.0" encoding="cp037"?><R/>',
                     b'<?xml version="1.0" encoding=cp037?><R/>',
                     b'<?xml version="1.0" encoding="utf-8" ' + b"x" * 600 + b"<R/>"):
            with self.subTest(data=data[:40]):
                self.assertTrue(g.refs_possible(data))

    def test_a_plain_document_is_counted_without_text_callbacks(self):
        data = b'<?xml version="1.0"?><R type="a">' + b"<I/> text\n" * 50 + b"</R>"
        self.assertFalse(g.refs_possible(data))
        self.assertEqual(g.scan_xml(data), reference_scan(data))


class FastScanMemory(unittest.TestCase):
    def peak(self, scan, data):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            result = scan(data)
            return tracemalloc.get_traced_memory()[1], result
        finally:
            tracemalloc.stop()

    def check_not_larger(self, data):
        ref_peak, want = self.peak(reference_scan, data)
        new_peak, got = self.peak(g.scan_xml, data)
        self.assertEqual(got, want)
        self.assertLessEqual(new_peak, ref_peak)

    def test_dense_document(self):
        self.check_not_larger(b"<r>" + b"".join(b'<i type="t%d" a="v"><c>x</c></i>' % (i % 50)
                                                 for i in range(20000)) + b"</r>")

    def test_dense_document_of_distinct_names(self):
        self.check_not_larger(b"<r>" + b"".join(b'<t%d a%d="$(v%d)" type="ty%d"/>' % (i, i, i, i)
                                                 for i in range(8000)) + b"</r>")

    def test_sparse_document_with_one_long_text(self):
        self.check_not_larger(b"<r>" + b"abc def " * 400_000 + b"</r>")

    def test_sparse_document_with_one_long_text_of_references(self):
        self.check_not_larger(b"<r>" + b"$(m) platform:/levels " * 150_000 + b"<!--c--></r>")

    def test_sparse_document_with_one_long_attribute(self):
        self.check_not_larger(b'<r a="' + b"abc def " * 300_000 + b'"/>')

    def test_deep_document(self):
        self.check_not_larger(b"<a>" * 5000 + b"</a>" * 5000)

    def test_malformed_dense_document(self):
        self.check_not_larger((b"<r>" + b"".join(b'<i type="t" a="$(v)"><c>x</c></i>' for i in range(8000)))[:-5])


if __name__ == "__main__":
    unittest.main()
