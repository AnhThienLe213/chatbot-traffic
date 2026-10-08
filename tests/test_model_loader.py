from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
import unittest

from traffic_law_assistant.config import Settings
from traffic_law_assistant.models import ModelBundle, ModelLoader


class ModelLoaderTests(unittest.TestCase):
    def test_concurrent_loads_share_the_same_model_bundle(self) -> None:
        factory_calls = 0
        factory_lock = Lock()
        expected_bundle = ModelBundle(
            embedder=object(),
            reranker=object(),
            tokenizer=object(),
            language_model=object(),
        )

        def factory(_: Settings) -> ModelBundle:
            nonlocal factory_calls
            with factory_lock:
                factory_calls += 1
            sleep(0.02)
            return expected_bundle

        loader = ModelLoader(Settings(), factory=factory)
        with ThreadPoolExecutor(max_workers=8) as executor:
            bundles = list(executor.map(lambda _: loader.load(), range(32)))

        self.assertEqual(factory_calls, 1)
        self.assertTrue(all(bundle is expected_bundle for bundle in bundles))


if __name__ == "__main__":
    unittest.main()
