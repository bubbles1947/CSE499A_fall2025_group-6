import api from "./api";

export interface LoginPayload {
  email: string;
  password: string;
}

export interface RegisterPayload {
  email: string;
  username: string;
  password: string;
}

export const loginUser = async (payload: LoginPayload) => {
  const res = await api.post("/auth/login", payload);
  return res.data; // { access_token, token_type }
};

export const registerUser = async (payload: RegisterPayload) => {
  const res = await api.post("/auth/register", payload);
  return res.data;
};

export const getMe = async () => {
  const res = await api.get("/auth/me");
  return res.data;
};
