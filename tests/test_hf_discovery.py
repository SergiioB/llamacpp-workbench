"""Tests for hf_discovery module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llama_webui.hf_discovery import (
    DiscoveredModel,
    GgufFile,
    HardwareProfile,
    _cache,
    _extract_quant,
    _extract_size_tier,
    _is_moe,
    get_gguf_files,
    rank_models,
    recommend_models,
    search_models,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear the in-memory HF API cache between tests."""
    _cache.clear()


class TestGgufFile:
    def test_quant_extraction_q4km(self) -> None:
        f = GgufFile(filename="qwen3-35b-Q4_K_M.gguf", size_bytes=8_000_000_000)
        assert f.quant_label == "Q4_K_M"
        assert f.quant_score == 10
        assert f.size_gb == 7.45

    def test_quant_extraction_q8(self) -> None:
        f = GgufFile(filename="model-Q8_0.gguf", size_bytes=16_000_000_000)
        assert f.quant_label == "Q8_0"
        assert f.quant_score == 6

    def test_quant_unknown(self) -> None:
        f = GgufFile(filename="model.gguf", size_bytes=5_000_000_000)
        assert f.quant_label == ""
        assert f.quant_score == 0


class TestHardwareProfile:
    def test_max_model_gb_from_ram(self) -> None:
        hw = HardwareProfile(total_ram_gb=23.0, max_model_gb=16.0)
        assert hw.max_model_gb == 16.0

    @patch("llama_webui.hf_discovery._detect_total_ram_gb", return_value=16.0)
    @patch("llama_webui.hf_discovery._get_machine", return_value="aarch64")
    @patch("llama_webui.hf_discovery._detect_gpu_backend", return_value="cpu")
    def test_from_system_16gb(self, mock_gpu: MagicMock, mock_machine: MagicMock, mock_ram: MagicMock) -> None:
        hw = HardwareProfile.from_system()
        assert hw.total_ram_gb == 16.0
        assert hw.max_model_gb == 9.0
        assert hw.is_arm is True
        assert hw.gpu_backend == "cpu"

    @patch("llama_webui.hf_discovery._detect_total_ram_gb", return_value=8.0)
    @patch("llama_webui.hf_discovery._get_machine", return_value="aarch64")
    @patch("llama_webui.hf_discovery._detect_gpu_backend", return_value="cpu")
    def test_from_system_low_ram_clamp(self, mock_gpu: MagicMock, mock_machine: MagicMock, mock_ram: MagicMock) -> None:
        hw = HardwareProfile.from_system()
        assert hw.max_model_gb == 4.0  # max(8-7, 4) = 4


class TestIsMoe:
    def test_a3b(self) -> None:
        assert _is_moe("Qwen3.6-35B-A3B") is True

    def test_a4b(self) -> None:
        assert _is_moe("gemma-4-26b-A4B-it") is True

    def test_dense(self) -> None:
        assert _is_moe("llama-3-8b") is False

    def test_moe_in_name(self) -> None:
        assert _is_moe("mixtral-moe-8x7b") is True


class TestExtractQuant:
    def test_q4km(self) -> None:
        assert _extract_quant("model-Q4_K_M.gguf") == "Q4_K_M"

    def test_q5ks(self) -> None:
        assert _extract_quant("model-Q5_K_S.gguf") == "Q5_K_S"

    def test_no_quant(self) -> None:
        assert _extract_quant("model.gguf") == ""


class TestExtractSizeTier:
    def test_35b(self) -> None:
        assert _extract_size_tier("Qwen3.6-35B-A3B") == 16

    def test_8b(self) -> None:
        assert _extract_size_tier("llama-3-8b") == 9

    def test_unknown(self) -> None:
        assert _extract_size_tier("tiny-model") == 1


class TestRankModels:
    @pytest.fixture()
    def hardware(self) -> HardwareProfile:
        return HardwareProfile(total_ram_gb=23.0, max_model_gb=16.0, gpu_backend="cpu", is_arm=True)

    @pytest.fixture()
    def mock_gguf_files(self) -> None:
        """Patch get_gguf_files to return predictable results."""
        with patch("llama_webui.hf_discovery.get_gguf_files") as mock:
            def side_effect(repo_id: str) -> list[GgufFile]:
                if "qwen-moe" in repo_id:
                    return [GgufFile("qwen-moe-Q4_K_M.gguf", 12 * 1024**3)]
                if "llama-8b" in repo_id:
                    return [GgufFile("llama-8b-Q4_K_M.gguf", 5 * 1024**3)]
                if "giant" in repo_id:
                    return [GgufFile("giant-Q4_K_M.gguf", 50 * 1024**3)]
                return []
            mock.side_effect = side_effect
            yield mock

    def test_moe_scores_higher_than_dense(
        self, hardware: HardwareProfile, mock_gguf_files: None
    ) -> None:
        raw = [
            {"id": "user/llama-8b", "downloads": 100_000, "tags": ["gguf"]},
            {"id": "user/qwen-moe", "downloads": 100_000, "tags": ["gguf"]},
        ]
        ranked = rank_models(raw, hardware)
        assert len(ranked) == 2
        assert ranked[0].is_moe is True
        assert ranked[0].rank_score > ranked[1].rank_score

    def test_too_large_model_doesnt_fit(
        self, hardware: HardwareProfile, mock_gguf_files: None
    ) -> None:
        raw = [
            {"id": "user/giant", "downloads": 10, "tags": ["gguf"]},
        ]
        ranked = rank_models(raw, hardware)
        assert len(ranked) == 1
        assert ranked[0].fits_hardware is False

    def test_arch_pref_moe_filters_dense(
        self, hardware: HardwareProfile, mock_gguf_files: None
    ) -> None:
        raw = [
            {"id": "user/llama-8b", "downloads": 100_000, "tags": ["gguf"]},
            {"id": "user/qwen-moe", "downloads": 100_000, "tags": ["gguf"]},
        ]
        ranked = rank_models(raw, hardware, arch_pref="moe")
        assert all(m.is_moe for m in ranked)

    def test_arch_pref_dense_filters_moe(
        self, hardware: HardwareProfile, mock_gguf_files: None
    ) -> None:
        raw = [
            {"id": "user/llama-8b", "downloads": 100_000, "tags": ["gguf"]},
            {"id": "user/qwen-moe", "downloads": 100_000, "tags": ["gguf"]},
        ]
        ranked = rank_models(raw, hardware, arch_pref="dense")
        assert all(not m.is_moe for m in ranked)


class TestSearchModels:
    @patch("llama_webui.hf_discovery._hf_api_request")
    def test_search_returns_list(self, mock_req: MagicMock) -> None:
        mock_req.return_value = [
            {"id": "user/model1", "downloads": 5000},
            {"id": "user/model2", "downloads": 3000},
        ]
        results = search_models(limit=10)
        assert len(results) == 2

    @patch("llama_webui.hf_discovery._hf_api_request")
    def test_search_returns_empty_on_failure(self, mock_req: MagicMock) -> None:
        mock_req.return_value = None
        results = search_models()
        assert results == []


class TestGetGgufFiles:
    @patch("llama_webui.hf_discovery._hf_api_request")
    def test_extracts_gguf_files(self, mock_req: MagicMock) -> None:
        mock_req.return_value = [
            {"path": "model-Q4_K_M.gguf", "size": 5_000_000_000, "type": "file"},
            {"path": "model-Q8_0.gguf", "size": 9_000_000_000, "type": "file"},
            {"path": "README.md", "size": 5000, "type": "file"},
        ]
        files = get_gguf_files("user/model")
        assert len(files) == 2
        assert files[0].quant_label == "Q4_K_M"
        assert files[1].quant_label == "Q8_0"

    @patch("llama_webui.hf_discovery._hf_api_request")
    def test_returns_empty_on_failure(self, mock_req: MagicMock) -> None:
        mock_req.return_value = None
        assert get_gguf_files("user/model") == []


class TestRecommendModels:
    @patch("llama_webui.hf_discovery.search_models")
    @patch("llama_webui.hf_discovery.get_gguf_files")
    def test_end_to_end(self, mock_files: MagicMock, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {"id": "google/gemma-4-26B-A4B-it-GGUF", "downloads": 50_000, "tags": ["gguf"]},
            {"id": "user/tiny-3b", "downloads": 1_000, "tags": ["gguf"]},
        ]
        mock_files.side_effect = lambda rid: [
            GgufFile("gemma-a4b-Q4_K_M.gguf", 8 * 1024**3),
        ] if "gemma" in rid else [
            GgufFile("tiny-Q4_K_M.gguf", 2 * 1024**3),
        ]

        hw = HardwareProfile(total_ram_gb=23.0, max_model_gb=16.0)
        results = recommend_models(hw, limit=5)
        assert len(results) <= 5
        assert results[0].fits_hardware is True
        assert results[0].is_moe is True
        assert results[0].rank_score > 0


class TestDiscoveredModelToDict:
    def test_to_dict_structure(self) -> None:
        gguf = GgufFile("model-Q4_K_M.gguf", 5_000_000_000)
        model = DiscoveredModel(
            repo_id="user/model",
            model_name="model",
            is_moe=True,
            gguf_files=[gguf],
            fits_hardware=True,
            rank_score=42,
            downloads=10_000,
            best_gguf=gguf,
        )
        d = model.to_dict()
        assert d["repo_id"] == "user/model"
        assert d["is_moe"] is True
        assert d["fits_hardware"] is True
        assert len(d["gguf_files"]) == 1
        assert d["best_gguf"]["quant"] == "Q4_K_M"
