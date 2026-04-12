from retrieval import retrieve
from generation import generate_answer

# ==========================================
# CLI
# ==========================================

def main():
    print("\n✅ system ready. please input the patient's information (input 'q' to exit):\n")

    while True:
        try:
            query = input("👨‍👩‍👧 the patient's question: ").strip()
            if query.lower() == "q":
                break

            print("\n📚 retrieving medical guide...")
            parent_nodes = retrieve(query, top_k=3)

            print(f"\n📦 Final returned parent nodes: {len(parent_nodes)}")

            print("\n💡 generating answer...")
            answer = generate_answer(query, parent_nodes)

            print("\n" + "=" * 60)
            print("🤖 the assistant's answer:\n")
            print(answer)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n👋 Bye.")
            break
        except Exception as e:
            print(f"\n❌ error: {e}")


if __name__ == "__main__":
    main()
