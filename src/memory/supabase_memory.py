import os
from typing import List, Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseMemory:
    def __init__(self):
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment variables")
            
        self.client: Client = create_client(self.url, self.key)
        self.table_name = "chat_history"

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None, thinking_time: float = None) -> Dict[str, Any]:
        """
        Add a message to the chat history.
        """
        if metadata is None:
            metadata = {}
            
        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata,
            "thinking_time": thinking_time
        }
        
        response = self.client.table(self.table_name).insert(data).execute()
        return response.data[0] if response.data else None

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a session.
        """
        response = self.client.table(self.table_name)\
            .select("*")\
            .eq("session_id", session_id)\
            .order("created_at", desc=False)\
            .limit(limit)\
            .execute()
            
        return response.data

    def clear_history(self, session_id: str) -> None:
        """
        Clear chat history for a session.
        """
        self.client.table(self.table_name).delete().eq("session_id", session_id).execute()

    def get_all_sessions(self) -> List[str]:
        """
        Retrieve all distinct session IDs.
        """
        # Note: Supabase doesn't support distinct() directly on select() easily in all client versions
        # but we can use a stored procedure or just fetch all and dedup in python for now if volume is low.
        # Better approach: Create a separate 'sessions' table or use a specific query.
        # For now, let's try to fetch distinct session_ids.
        
        # Using a raw query or rpc is better, but let's try a simple approach first.
        # Since we don't have a sessions table, we have to query chat_history.
        
        response = self.client.table(self.table_name).select("session_id").execute()
        if response.data:
            # Dedup and return
            return list(set(item["session_id"] for item in response.data))
        return []
