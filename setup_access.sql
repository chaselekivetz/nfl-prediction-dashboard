create table if not exists public.app_users (
    email text primary key,
    active boolean not null default true,
    invited_at timestamptz not null default now(),
    invited_by text
);

alter table public.app_users enable row level security;

-- No public RLS policies are intentionally created.
-- The Streamlit server accesses this table only with the Supabase service-role key,
-- stored in Streamlit Secrets and never committed to GitHub.
