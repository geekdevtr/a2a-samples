import asyncio
import uuid
import httpx
import streamlit as st
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest


async def send_to_host(host_url: str, user_text: str):
    async with httpx.AsyncClient(timeout=60) as http_client:
        # 1) Discover the host's AgentCard
        resolver = A2ACardResolver(http_client, host_url)
        card = await resolver.get_agent_card()

        # 2) Build A2A message payload with required messageId
        message_id = uuid.uuid4().hex

        payload = {
            "message": {
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": user_text,
                    }
                ],
                "messageId": message_id,
            }
        }

        params = MessageSendParams.model_validate(payload)
        request = SendMessageRequest(id=message_id, params=params)

        # 3) Create client bound to this host and send message
        client = A2AClient(http_client, card, url=host_url)
        response = await client.send_message(request)
        return card, response



def main():
    st.set_page_config(page_title="Local A2A Inspector", layout="wide")
    st.title("Local A2A Inspector (Host + Remote Agents)")

    default_host = "http://localhost:8083"
    host_url = st.text_input("Host URL", value=default_host)

    user_text = st.text_area(
        "Message to Host Agent",
        value="Please use your remote agents to give me some weather info and accommodation suggestions.",
        height=200,
    )

    if st.button("Send to Host"):
        if not user_text.strip():
            st.warning("Please enter a message.")
            return

        with st.spinner("Calling host agent via A2A..."):
            try:
                card, response = asyncio.run(send_to_host(host_url, user_text))
            except Exception as e:
                st.error(f"Error calling host: {e}")
                return

        st.subheader("Resolved Host AgentCard")
        st.json(card.model_dump(exclude_none=True))

        st.subheader("Raw Response from Host")
        st.json(response.model_dump(exclude_none=True))


if __name__ == "__main__":
    main()
