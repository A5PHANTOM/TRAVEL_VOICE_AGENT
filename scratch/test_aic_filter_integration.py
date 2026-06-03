import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the app and pipecat modules are in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
sys.path.insert(1, os.path.join(project_root, "pipecat/src"))

class TestAICFilterIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Clear any environment variables for the test
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("app.main.AIC_FILTER_AVAILABLE", True)
    @patch("app.main.AICFilter", create=True)
    @patch("app.main._get_llm_api_key", return_value="dummy_key")
    async def test_run_bot_with_aic_filter(self, mock_get_key, mock_aic_filter_class):
        # Setup mock AICFilter instance
        mock_filter_instance = MagicMock()
        mock_aic_filter_class.return_value = mock_filter_instance
        
        # Import run_bot dynamically and patch module-level env cache
        import app.main as main_mod
        main_mod.aic_license_key = "test_license_key"
        main_mod.aic_model_id = "test_model_id"
        
        # Mock transport and connection
        mock_conn = MagicMock()
        mock_transport_class = MagicMock()
        
        with patch("app.main._run_agent", new_callable=AsyncMock) as mock_run_agent, \
             patch("app.main.SmallWebRTCTransport", return_value=mock_transport_class) as mock_webrtc_transport_class:
            
            # Execute run_bot
            await main_mod.run_bot(mock_conn)
            
            # Verify AICFilter was initialized with correct arguments
            mock_aic_filter_class.assert_called_once_with(
                license_key="test_license_key",
                model_id="test_model_id",
                enhancement_level=1.0,
            )
            
            # Verify transport parameters contained the filter
            args, kwargs = mock_webrtc_transport_class.call_args
            params = kwargs.get("params") or args[1]
            self.assertEqual(params.audio_in_filter, mock_filter_instance)
            
            # Verify agent was run with the transport and the filter
            mock_run_agent.assert_called_once_with(mock_transport_class, aic_filter=mock_filter_instance)

    @patch("app.main.AIC_FILTER_AVAILABLE", True)
    @patch("app.main.AICFilter", create=True)
    @patch("app.main._get_llm_api_key", return_value="dummy_key")
    async def test_run_bot_fallback_no_key(self, mock_get_key, mock_aic_filter_class):
        import app.main as main_mod
        main_mod.aic_license_key = ""

        mock_conn = MagicMock()
        mock_transport_class = MagicMock()

        with patch("app.main._run_agent", new_callable=AsyncMock) as mock_run_agent, \
             patch("app.main.SmallWebRTCTransport", return_value=mock_transport_class):
            
            # Executing run_bot should now raise RuntimeError because no key is set
            with self.assertRaises(RuntimeError) as context:
                await main_mod.run_bot(mock_conn)
            self.assertIn("AIC_LICENSE_KEY is not set", str(context.exception))

    @patch("app.main._get_llm_api_key", return_value="dummy_key")
    async def test_run_agent_vad_fallback(self, mock_get_key):
        import app.main as main_mod
        
        # Setup mock transport and context aggregator
        mock_transport = MagicMock()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("app.main.GroqLLMService"), \
             patch("app.main.DeepgramSTTService"), \
             patch("app.main.DeepgramHttpTTSService"), \
             patch("app.main.LLMContextAggregatorPair", return_value=(MagicMock(), MagicMock())) as mock_agg_pair, \
             patch("app.main.PipelineRunner") as mock_runner:
            
            mock_runner.return_value.run = AsyncMock()
            
            # Run agent with aic_filter = None, which should use SileroVADAnalyzer directly
            await main_mod._run_agent(mock_transport, aic_filter=None)
            
            # Verify Silero VAD was passed to aggregator
            args, kwargs = mock_agg_pair.call_args
            user_params = kwargs.get("user_params")
            self.assertIsInstance(user_params.vad_analyzer, main_mod.SileroVADAnalyzer)

    @patch("app.main._get_llm_api_key", return_value="dummy_key")
    @patch("app.main.StartupProtectedAICVADAnalyzer")
    async def test_run_agent_vad_aic(self, mock_protected_vad_class, mock_get_key):
        import app.main as main_mod
        
        # Manually set license key to trigger AIC path
        main_mod.aic_license_key = "test_license"
        
        # Setup mock transport, session, and context aggregator
        mock_transport = MagicMock()
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        mock_aic_filter = MagicMock()
        mock_protected_vad_instance = MagicMock()
        mock_protected_vad_class.return_value = mock_protected_vad_instance
        
        with patch("aiohttp.ClientSession", return_value=mock_session), \
             patch("app.main.GroqLLMService"), \
             patch("app.main.DeepgramSTTService"), \
             patch("app.main.DeepgramHttpTTSService"), \
             patch("app.main.LLMContextAggregatorPair", return_value=(MagicMock(), MagicMock())) as mock_agg_pair, \
             patch("app.main.PipelineRunner") as mock_runner:
            
            mock_runner.return_value.run = AsyncMock()
            
            # Run agent with mock aic_filter
            await main_mod._run_agent(mock_transport, aic_filter=mock_aic_filter)
            
            # Verify the StartupProtectedAICVADAnalyzer was created with correct args
            mock_protected_vad_class.assert_called_once()
            args, kwargs = mock_protected_vad_class.call_args
            self.assertEqual(kwargs.get("speech_hold_duration"), 0.6)
            self.assertEqual(kwargs.get("minimum_speech_duration"), 0.15)
            self.assertEqual(kwargs.get("sensitivity"), 5.3)
            
            # Verify FallbackVADAnalyzer wrapped it and was passed to aggregator
            args, kwargs = mock_agg_pair.call_args
            user_params = kwargs.get("user_params")
            self.assertIsInstance(user_params.vad_analyzer, main_mod.FallbackVADAnalyzer)
            self.assertEqual(user_params.vad_analyzer.primary_vad, mock_protected_vad_instance)

if __name__ == "__main__":
    unittest.main()
