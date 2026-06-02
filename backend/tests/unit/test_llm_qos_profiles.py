"""Tests for the 'local_only' Model QoS profile (fully-local self-host).

The 'local_only' profile is opt-in via MODEL_QOS=local_only. It must:
  - route every cloud feature to a model served by the local LiteLLM proxy
  - never reference the 'gemini' provider (no direct Google AI Studio calls)
  - never reference the 'openrouter' provider (no OpenRouter cloud calls)
  - cover the full feature set of the 'premium' profile (no dropped features)

The only remaining cloud dependency is web_search → sonar-pro (Perplexity),
which has no local equivalent yet and is documented as 'still cloud OR
disable via env'.
"""

import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Pre-mock heavy deps before importing clients.py (Firebase, Firestore, DB).
# ---------------------------------------------------------------------------
_HEAVY_MOCKS = {
    'firebase_admin': MagicMock(),
    'firebase_admin.firestore': MagicMock(),
    'google.cloud.firestore': MagicMock(),
    'google.cloud.firestore_v1': MagicMock(),
    'google.cloud.firestore_v1.base_query': MagicMock(),
    'database': MagicMock(),
    'database._client': MagicMock(),
    'database.llm_usage': MagicMock(),
}
for _mod, _mock in _HEAVY_MOCKS.items():
    sys.modules.setdefault(_mod, _mock)

os.environ.setdefault('OPENAI_API_KEY', 'sk-test-fake-key-for-unit-tests')
os.environ.setdefault('ANTHROPIC_API_KEY', 'sk-ant-test-fake-key')

from utils.llm.clients import MODEL_QOS_PROFILES  # noqa: E402

# Models confirmed locally hosted on the rtx6000 LiteLLM proxy
# (http://10.0.60.48:4000). Verified via /v1/models list + latency probes
# during the scout phase. This set MUST NOT include any cloud-only model.
_LOCAL_MODELS_ON_LITELLM = {
    'qwen3.6-35b-a3b',
    'sonar-pro',  # Perplexity — irreplaceable cloud dep, documented exception
}

# The set of cloud providers that local_only must NEVER reference.
_FORBIDDEN_PROVIDERS_FOR_LOCAL_ONLY = {'gemini', 'openrouter'}


class TestLocalOnlyProfile:
    def test_local_only_profile_exists(self):
        """The 'local_only' profile must be registered in MODEL_QOS_PROFILES."""
        assert 'local_only' in MODEL_QOS_PROFILES
        assert isinstance(MODEL_QOS_PROFILES['local_only'], dict)
        assert len(MODEL_QOS_PROFILES['local_only']) > 0

    def test_local_only_uses_no_gemini_provider(self):
        """No feature in local_only may route to provider='gemini' (direct Google AI Studio)."""
        profile = MODEL_QOS_PROFILES['local_only']
        gemini_features = [feat for feat, (_m, prov) in profile.items() if prov == 'gemini']
        assert not gemini_features, f"local_only must not use 'gemini' provider; offenders: {gemini_features}"

    def test_local_only_uses_no_openrouter_provider(self):
        """No feature in local_only may route to provider='openrouter'."""
        profile = MODEL_QOS_PROFILES['local_only']
        or_features = [feat for feat, (_m, prov) in profile.items() if prov == 'openrouter']
        assert not or_features, f"local_only must not use 'openrouter' provider; offenders: {or_features}"

    def test_local_only_forbidden_providers_absent(self):
        """Sanity wrapper: providers reaching third-party clouds are absent."""
        profile = MODEL_QOS_PROFILES['local_only']
        providers = {prov for _model, prov in profile.values()}
        leaked = providers & _FORBIDDEN_PROVIDERS_FOR_LOCAL_ONLY
        assert not leaked, f"local_only leaks to cloud providers: {leaked}"

    def test_local_only_models_are_all_in_local_set(self):
        """Every model referenced must be hosted on the local LiteLLM proxy."""
        profile = MODEL_QOS_PROFILES['local_only']
        for feature, (model, _provider) in profile.items():
            assert model in _LOCAL_MODELS_ON_LITELLM, (
                f"local_only feature '{feature}' uses model '{model}' which is not in the "
                f"local-on-LiteLLM set {sorted(_LOCAL_MODELS_ON_LITELLM)}"
            )

    def test_local_only_covers_all_tasks_in_premium(self):
        """No feature from premium may be silently dropped in local_only."""
        premium_features = set(MODEL_QOS_PROFILES['premium'].keys())
        local_features = set(MODEL_QOS_PROFILES['local_only'].keys())
        missing = premium_features - local_features
        extra = local_features - premium_features
        assert not missing, f"local_only is missing features present in premium: {missing}"
        assert not extra, f"local_only has features not in premium: {extra}"

    def test_get_qos_profile_local_only_works(self):
        """Accessing the local_only profile through MODEL_QOS_PROFILES yields the expected dict shape."""
        profile = MODEL_QOS_PROFILES.get('local_only')
        assert profile is not None
        # Each value must be a 2-tuple of (model: str, provider: str)
        for feature, value in profile.items():
            assert isinstance(value, tuple), f"{feature} value should be tuple"
            assert len(value) == 2, f"{feature} value should be (model, provider)"
            model, provider = value
            assert isinstance(model, str) and model, f"{feature} model must be non-empty string"
            assert isinstance(provider, str) and provider, f"{feature} provider must be non-empty string"

    def test_local_only_chat_agent_routes_to_anthropic_endpoint(self):
        """chat_agent must keep provider='anthropic' so the anthropic_client tool-use path works.

        The LiteLLM Anthropic-compat endpoint at ANTHROPIC_BASE_URL serves qwen3.6-35b-a3b on
        /v1/messages, verified by curl probe during scout phase.
        """
        profile = MODEL_QOS_PROFILES['local_only']
        model, provider = profile['chat_agent']
        assert provider == 'anthropic', f"chat_agent must stay anthropic, got {provider}"
        assert model in _LOCAL_MODELS_ON_LITELLM, f"chat_agent model {model} must be local"

    def test_local_only_web_search_documented_cloud_exception(self):
        """web_search is the documented cloud exception (no local Perplexica shim yet)."""
        profile = MODEL_QOS_PROFILES['local_only']
        model, provider = profile['web_search']
        assert provider == 'perplexity', "web_search remains on perplexity until sonar-local shim ships"
        assert model == 'sonar-pro'

    def test_local_only_eliminates_premium_gemini_tasks(self):
        """The 5 Gemini-cloud tasks from premium must NOT use Gemini in local_only."""
        premium_gemini_tasks = {feat for feat, (_m, prov) in MODEL_QOS_PROFILES['premium'].items() if prov == 'gemini'}
        # Sanity check: premium really does have these Gemini tasks
        assert premium_gemini_tasks, "Premium should have gemini-routed features"
        local = MODEL_QOS_PROFILES['local_only']
        for feature in premium_gemini_tasks:
            _model, provider = local[feature]
            assert provider != 'gemini', f"{feature} still routes to gemini in local_only"

    def test_local_only_eliminates_openrouter_wrapped_analysis(self):
        """wrapped_analysis (only OpenRouter task) must be local in local_only."""
        local = MODEL_QOS_PROFILES['local_only']
        model, provider = local['wrapped_analysis']
        assert provider != 'openrouter', f"wrapped_analysis still on openrouter; got provider={provider}"
        assert model in _LOCAL_MODELS_ON_LITELLM
