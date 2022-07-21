from icevision.engines.lightning import LightningModelAdapter


class MockMetric:
    def __init__(self):
        self.name = "MockMetric"

    def accumulate(self):
        pass

    def finalize(sefl):
        return {"metric_a": 1, "metric_b": 2}


class DummLightningModelAdapter(LightningModelAdapter):
    pass


def test_finalze_metrics_reports_metrics_correctly(mocker):
    mocker.patch(
        "icevision.engines.lightning.lightning_model_adapter.LightningModelAdapter.log"
    )

    adapter = DummLightningModelAdapter([MockMetric()], [("metric_a", "a")])
    adapter.finalize_metrics()

    adapter.log.assert_any_call("a", 1, prog_bar=True)
    adapter.log.assert_any_call("MockMetric/metric_a", 1)
    adapter.log.assert_any_call("MockMetric/metric_b", 2)
