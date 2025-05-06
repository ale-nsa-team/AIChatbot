import os
import logging
import json
from datetime import datetime
import pytz
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse  # Import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage
from search import get_intent_chain, convert_configuration_chain, qa_chain  # import your existing RAG setup

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

mock_intent_response = AIMessage(
    content='{ "result": "[{\\"domain\\": \\"external_knowledge\\", \\"sub_intent\\": \\"convert_configuration\\", \\"feature\\": \\"ospf\\", \\"entities\\": {}}]" }'
)


class UserInput(BaseModel):
    user_input: str


@app.get("/")
async def get_index():
    # Return the index.html file from the static folder
    file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(file_path, "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# @app.post("/process_input")
# def chat(user_input: UserInput):
#     try:
#         responses = qa_chain.invoke({"query": user_input.user_input})
#         return {"responses": responses}
#     except Exception as e:
#         return {"error": str(e)}

@app.post("/process_input")
def chat(user_input: UserInput):
    MOCK = False
    try:
        question = user_input.user_input

        if MOCK:
            question = """
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

            """  # Your mock Cisco config text
            intent_result_json = json.loads(json.loads(mock_intent_response.content)["result"])
        else:
            intent_res = get_intent_chain.invoke({"question": question})
            intent_result_json = json.loads(json.loads(intent_res.content)["result"])

        domain = intent_result_json[0]["domain"]
        sub_intent = intent_result_json[0]["sub_intent"]

        intent = {"domain": domain, "sub_intent": sub_intent}
        response_payload = {"message": "", "answer": "", "timestamp": None, "timezone": None}

        if domain == "external_knowledge":
            if sub_intent == "convert_configuration":
                timezone = "Asia/Singapore"
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz).strftime("%d %b, %Y, %H:%M:%S")

                response = convert_configuration_chain.invoke({"question": question})
                response_payload = {
                    "message": f"""
                    Here is the response to your question. \n
                    ! AI Generated Alcatel AOS configuration at {current_time} in {timezone}.""",
                    "answer": response.content,
                    "timestamp": current_time,
                    "timezone": timezone
                }
            else:
                response = qa_chain.invoke({"query": question})
                response_payload = {
                    "message": "Here is the response to your question.\n",
                    "answer": response["result"],
                    "timestamp": None,
                    "timezone": None
                }

        return {
            "status": "success",
            "intent": intent,
            "response": response_payload,
            "error": None
        }

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return {
            "status": "error",
            "intent": None,
            "response": None,
            "error": str(e)
        }

