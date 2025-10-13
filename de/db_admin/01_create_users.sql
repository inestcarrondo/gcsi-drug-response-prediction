-- Create users and roles as required by the challenge

-- Create ds_user_role with READ-ONLY permissions on public schema
CREATE ROLE ds_user_role;
GRANT USAGE ON SCHEMA public TO ds_user_role;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ds_user_role;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO ds_user_role;

-- Ensure future tables also get SELECT permissions
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ds_user_role;

-- Create ds_user with ds_user_role
CREATE USER ds_user WITH PASSWORD 'ds_user';
GRANT ds_user_role TO ds_user;

-- admin user already exists as POSTGRES_USER, just ensure full permissions
-- (admin user is created automatically by PostgreSQL with superuser privileges)