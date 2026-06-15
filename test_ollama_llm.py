"""
Test script for Local LLM (Ollama) integration.
Demonstrates end-to-end usage of LocalLLMHandler with a sample query.
"""

import os
from cloud_llm import LocalLLMHandler


def test_local_llm():
    """Test LocalLLMHandler with a sample query."""
    
    print("🚀 Initializing LocalLLMHandler (Ollama)...")
    
    try:
        # Initialize the handler
        llm = LocalLLMHandler(model="mistral")
        print("✅ LocalLLMHandler initialized successfully")
        print("✅ Ollama is running and accessible")
        
        # Test token counting
        sample_text = "This is a sample text to test token counting."
        token_count = llm.count_tokens(sample_text)
        print(f"📊 Token count for sample text: {token_count}")
        
        # Test response generation with sample context
        query = "What is the capital of France?"
        context = """France is a country in Western Europe. 
Its capital city is Paris, which is known for the Eiffel Tower, 
the Louvre Museum, and its rich cultural heritage. 
Paris has been the political and cultural center of France for centuries."""
        
        print(f"\n🔍 Query: {query}")
        print(f"📄 Context: {context[:100]}...")
        
        print("\n⏳ Generating response with Ollama...")
        result = llm.generate_response(query, context)
        
        print("\n✅ Response generated successfully!")
        print(f"\n📝 Response:\n{result['response']}")
        print(f"\n📊 Usage Statistics:")
        print(f"  - Input tokens (estimated): {result['input_tokens']}")
        print(f"  - Output tokens (estimated): {result['output_tokens']}")
        print(f"  - Cost: ${result['cost']:.2f} (FREE! 🎉)")
        print(f"  - Model: {result['model']}")
        print(f"  - Timestamp: {result['timestamp']}")
        
        # Get overall usage stats
        stats = llm.get_usage_stats()
        print(f"\n📈 Cumulative Usage Statistics:")
        print(f"  - Total requests: {stats['total_requests']}")
        print(f"  - Total input tokens (estimated): {stats['total_input_tokens']}")
        print(f"  - Total output tokens (estimated): {stats['total_output_tokens']}")
        print(f"  - Total cost: ${stats['cost']:.2f} (FREE! 🎉)")
        print(f"  - Status: {stats['status']}")
        
        print("\n✅ All tests passed!")
        
    except ConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("\n🔧 Fix:")
        print("   1. Make sure Ollama is installed from https://ollama.ai")
        print("   2. Run: ollama pull mistral")
        print("   3. Run: ollama run mistral")
        print("   4. Then run this test again")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Local LLM (Ollama) Integration Test")
    print("=" * 60)
    print()
    
    test_local_llm()
    
    print()
    print("=" * 60)