from agent_flows import test_agents

if __name__ == '__main__':
    # Query related to Python lecture content
    query = 'What are the main topics covered in this Python lecture?'
    print("\nQuery:", query)
    response = test_agents(query)
    print("\nResponse:", response) 