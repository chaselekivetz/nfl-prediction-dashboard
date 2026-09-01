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


-- Required when "Automatically expose new tables" is disabled in Supabase.
-- The server-side Secret key maps to the service_role database role.
grant usage on schema public to service_role;
grant select, insert, update on table public.app_users to service_role;
