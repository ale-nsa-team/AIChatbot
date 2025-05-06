import os
import json
import logging
from datetime import datetime
import pytz
# from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY

# Setup logging
#logging.basicConfig(level=logging.INFO)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

# Load FAISS vectorstore
index_path = "faiss_index"
if not os.path.exists(index_path):
    raise FileNotFoundError(f"Index path not found: {index_path}")

vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# LLM setup using LangChain wrapper
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY, 
    model="gpt-4o", 
    temperature=0.7
    )

# Custom prompt
intent_template = """
You are a network assistant. Your job is to classify user input into one of four domains:
1. live_network: Real-time actions on a network like configuration, monitoring, troubleshooting, or maintenance.
2. external_knowledge: Questions that require looking up documentation, vendor KBs, or manuals.
3. hybrid_intelligence: Analytics, audits, optimizations, or compliance checks.
4. automation: Task that needs to be performed by multiple AI agents.

Below are examples to guide you:

---

Example 1:
Input: "Configure VLAN 10 on all switches."
Domain: live_network
sub_intent: config_vlan

Example 2:
Input: "What is the maximum number of OSPF routes supported by OmniSwitch 9900?"
Domain: external_knowledge
sub_intent: specification

Example 3:
Input: "Audit all switch configurations to ensure they comply with security policies."
Domain: hybrid_intelligence
sub_intent: audit


Example 4:
Input: "Schedule firmware upgrades and backup all configurations across all data centers."
Domain: automation
sub_intent: firmware


Example 4:
Input: "Help me convert Cisco configuration to Alcatel configuration"
Domain: external_knowledge
sub_intent: convert_configuration


---

You must respond with a **JSON object** with the key `"result"` and a **JSON-formatted string** as its value.

Example output:
{{ 
  "result": "[{{\\"domain\\": \\"live_network\\", \\"sub_intent\\": \\"config\\", \\"feature\\": \\"ospf\\", \\"entities\\": {{\\"area\\": \\"0.0.0.0\\"}}}}]"
}}

Respond accordingly:
- Only respond with the object described.
- Do not explain, comment, or add extra formatting.
"""

# Custom prompt
convert_configuration_template = """
You are a network assistant. Your job is to convert ***Cisco <IOS> Configuration*** to ***Alcatel <AOS> Configuration***.
In the context below, it shows you the cisco configuration and the respective alcatel configuration.

Context:
!
-- Cisco <VLAN> configuration ---
!
vlan 1
   name "vlan_1"
!
vlan 2
   name "vlan_2"
!
-- Cisco <IP Interface & VLAN Membership> configuration ---
!
interface vlan 1
  ip address 192.168.1.1 255.255.255.0
  ip helper-address 192.168.2.254
!
interface vlan 2
  ip address 192.168.2.1 255.255.255.0
  ip helper-address 192.168.2.254
!
interface GigibitEthernet 1/1/1
   switchport access vlan 1
!
interface GigibitEthernet 1/1/2
   switchport access vlan 2
!

-- Alcatel <VLAN> configuration --- 
!
vlan 1 name "vlan_1"
vlan 2 name "vlan_2"
!
-- Alcatel <Port Membership> configuration --- 
!
vlan 1 member port 1/1/1 untagged
vlan 2 member port 1/1/2 untagged
!
-- Alcatel <IP Interface> configuration --- 
!
ip interface "int_vlan_1" address 192.168.1.1 mask 255.255.255.0 vlan 1
ip interface "int_vlan_2" address 192.168.2.1 mask 255.255.255.0 vlan 2
!



# Instructions
- Answer strictly based on context.
- If the answer is not in the example, say: "!  Our documentation does not contain the relevant information."
- You are to group similar commands together, separate the group with '!' and return the response in the form of a str.
- You MUST return in the following order: VLAN, Port membership, IP interface.

Example:
!
! VLAN configuration
!
vlan 1 name "vlan_1"
!
! Port Membership
!
vlan 1 member port 1/1/1
!
! IP Interface
!
ip interface "int_vlan_1" ip address 10.1.1.1 mask 255.255.255.0 vlan 1
!
"""

system_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(intent_template),
    HumanMessagePromptTemplate.from_template("User Query: {question}")
])

config_conversion_system_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(convert_configuration_template),
    HumanMessagePromptTemplate.from_template("User Query: {question}")
])



prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""

# Identity    
You are a helpful and technically accurate assistant specializing in networking.
You must strictly answer the user query using ONLY the documentation provided below.

Context:
{context}

User Query:
{question}

# Example:
*****************************
Question: How to configure ospf?

Answer:
To Configure OSPF, follow these steps:
1.	**Prepare the Routers**. Create VLANs on each router, add an IP interface to the VLAN, assign a port to the VLAN, and assign a router identification number.  For example, on Router 1:  (LFH e steps are all from Advanced Routing Guide pg. 48 or page 83 since the information is 'duplicated' in the AR guide. Interestingly, it omits the creation of vlan 10 and assigning it to an interface) (Also, it omits config of router 2 and 3)
-> vlan 31
-> ip  interface vlan-31 vlan 31 address 31.0.1.1 mask 255.0.0.0
-> vlan 31 members ports 2/1
-> vlan 12
-> ip interface vlan-12 vlan 12 address 12.0.0.1 mask 255.0.0.0
-> vlan 12 members port 2/2

2. **Enable OSPF**: Load and enable OSPF on each router:  
-> ip load ospf
-> ip ospf admin-state enable

3. **Create the OSPF Area**: Create the OSPF area (e.g., area 0.0.0.1):  
-> ip ospf area 0.0.0.1

4. **Configure OSPF Interfaces**: Create and enable OSPF interfaces, assigning them to the area created:  
For Router 1:
-> ip ospf interface vlan-31
-> ip ospf interface vlan-31 area 0.0.0.0
-> ip ospf interface vlan-31 admin-state enable
-> ip ospf interface vlan-12
-> ip ospf interface vlan-12 area 0.0.0.0
-> ip ospf interface vlan-12 admin-state enable
-> ip ospf interface vlan-10
-> ip ospf interface vlan-10 area 0.0.0.1
-> ip ospf interface vlan-10 admin-state enable 

5. **(Optional) Configure BFD**: if using BFD for faster failure detection, register OSPF with BFD 
-> ip ospf bfd-state enable

6. **(Optional) Enable BFD on Interfaces**: 
-> ip ospf interface vlan-10 bfd-state enable

7. **Verify Configuration**: Use the following commands to verify the OSPF configuration: 
-> show ip ospf
-> who ip ospf interface

**Sources**
- [AOS 8.9 R02 Advanced Routing Guide.pdf, Page 3, 4, 13-14, 18-19, 21, 29-30, 34]
- [AOS 8.9 R02 Network Configuration Guide.pdf, Page 613, 626]

*****************************

# Instructions
- Answer strictly based on context.
- If additional information is available, show the details as well.
- If the answer is not in the context, say: "The documentation does not contain that information."
- At the end, include a **Sources** list based on the metadata of the context documents.
"""
)

# Create a mock AIMessage response
mock_intent_response = AIMessage(
    content='{ "result": "[{\\"domain\\": \\"external_knowledge\\", \\"sub_intent\\": \\"convert_configuration\\", \\"feature\\": \\"ospf\\", \\"entities\\": {}}]" }'
)


# Create the QA Chain
get_intent_chain = system_prompt | llm

# Create the QA Chain
convert_configuration_chain = config_conversion_system_prompt | llm

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)


# Main Loop
if __name__ == "__main__":

    MOCK = True

    while True:
        user_input = input("\n💬 What would you like to do today? (type 'q' to quit): ").strip()
        if user_input.lower() in ["q", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            if MOCK:
                user_input = """
                Help me convert the following Cisco IOS configuration to Alcatel AOS configuration.

                -- Cisco configuration ---
                !
                vlan 1
                  name "Finance"
                vlan 2
                  name "HR"
                !
                interface vlan 1
                ip address 10.1.1.1 255.255.255.0
                ip helper-address 192.168.2.254
                !
                interface vlan 2
                ip address 10.1.2.1 255.255.255.0
                ip helper-address 192.168.2.254
                !
                interface GigibitEthernet 1/1/1
                switchport access vlan 1
                !
                interface GigibitEthernet 1/1/2
                switchport access vlan 2
                !                
                """
                intent_result_json = json.loads(json.loads(mock_intent_response.content)["result"])
            else:
                intent_res = get_intent_chain.invoke({"question": user_input})
                intent_result_json = json.loads(json.loads(intent_res.content)["result"])               

            intent_domain = intent_result_json[0]["domain"]
            intent_sub_intent = intent_result_json[0]["sub_intent"]
 
            if intent_domain in ['external_knowledge']:
                if intent_sub_intent in ['convert_configuration']:
                    timezone = "Asia/Singapore"
                    tz = pytz.timezone(timezone)
                    current_time = datetime.now(tz).strftime("%d %b, %Y, %H:%M:%S")                    
                    response = convert_configuration_chain.invoke({"question": user_input})
                    print("\nAnswer:\n")
                    print(f"! AI Generated Alcatel AOS configuration at {current_time} in {timezone}.")
                    print(response.content)
                else:
                    response = qa_chain.invoke({"query": user_input})
                    print(f"\nQuestion: {response['query']}")
                    print("\nAnswer:\n")
                    print(response['result'])


        except Exception as e:
            logging.error(f"Error occurred: {e}")
