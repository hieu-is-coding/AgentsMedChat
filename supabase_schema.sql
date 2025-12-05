-- Create a table to store chat history
create table if not exists chat_history (
  id uuid default gen_random_uuid() primary key,
  session_id text not null,
  role text not null,
  content text not null,
  metadata jsonb default '{}'::jsonb,
  thinking_time float,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null
);

-- Create an index on session_id for faster retrieval
create index if not exists idx_chat_history_session_id on chat_history(session_id);

-- Enable Row Level Security (RLS)
alter table chat_history enable row level security;

-- Policy to allow read access to everyone (modify as needed for production)
create policy "Allow public read access"
on chat_history
for select
to public
using (true);

-- Policy to allow insert access to everyone (modify as needed for production)
to public
with check (true);
