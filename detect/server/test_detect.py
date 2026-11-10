import unittest

from detect import boxes_from_results, decode_image, queries_of, threshold_of, wire_only


class DetectTests(unittest.TestCase):
    def test_queries_accept_labels_and_text(self):
        self.assertEqual(queries_of({"queries": ["cup", " "] }), ["cup"])
        self.assertEqual(queries_of({"labels": ["person"]}), ["person"])
        self.assertEqual(queries_of({"text": "mug"}), ["mug"])
        self.assertEqual(queries_of({}), [])

    def test_threshold_clamps(self):
        self.assertEqual(threshold_of({"threshold": 2}), 1.0)
        self.assertEqual(threshold_of({"threshold": "nope"}), 0.1)

    def test_decode_data_url(self):
        raw = decode_image("data:image/png;base64,aGk=")
        self.assertEqual(raw, b"hi")

    def test_boxes_use_query_names(self):
        class T(list):
            pass

        got = boxes_from_results(
            {"boxes": [[1, 2, 3, 4]], "scores": [0.91234], "labels": [0]},
            ["cup"],
        )
        self.assertEqual(got, [{"label": "cup", "score": 0.9123, "box": [1.0, 2.0, 3.0, 4.0]}])

    def test_wire_only_has_no_boxes(self):
        body = wire_only("missing")
        self.assertEqual(body["status"], "wire_only")
        self.assertEqual(body["detections"], [])


if __name__ == "__main__":
    unittest.main()
