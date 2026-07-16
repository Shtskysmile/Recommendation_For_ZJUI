from app.plugin_loader import PluginLoader


def test_plugin_loader_resolves_fallback_manifests(test_settings):
    loader = PluginLoader(test_settings.project_root)

    algorithms = {plugin.key for plugin in loader.list_algorithm_plugins("tabular_classification")}
    profile = loader.get_profile_for_task_type("tabular_classification")

    assert "risk_score" in algorithms
    assert "majority_class" in algorithms
    assert profile.key == "tabular_classification"
