import { create } from "zustand";
import AsyncStorage from "@react-native-async-storage/async-storage";
import api from "../services/api";

interface User {
  id: string;
  email: string;
  username: string;
  created_at: string;
}

interface AuthState {
  token: string | null;
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  loadToken: () => Promise<void>;
  fetchUser: () => Promise<void>;
}

const useAuthStore = create<AuthState>((set, get) => ({
  token: null,
  user: null,
  isLoading: true,
  isAuthenticated: false,

  login: async (email, password) => {
    console.log("[Store/login] POST /auth/login →", { email, password });
    const res = await api.post("/auth/login", { email, password });
    console.log("[Store/login] Response ←", res.status, res.data);
    const { access_token } = res.data;
    await AsyncStorage.setItem("token", access_token);
    api.defaults.headers.common["Authorization"] = `Bearer ${access_token}`;
    set({ token: access_token, isAuthenticated: true });
    await get().fetchUser();
  },

  register: async (email, username, password) => {
    await api.post("/auth/register", { email, username, password });
    // Auto-login after registration
    await get().login(email, password);
  },

  logout: async () => {
    await AsyncStorage.removeItem("token");
    delete api.defaults.headers.common["Authorization"];
    set({ token: null, user: null, isAuthenticated: false });
  },

  loadToken: async () => {
    try {
      const token = await AsyncStorage.getItem("token");
      if (token) {
        api.defaults.headers.common["Authorization"] = `Bearer ${token}`;
        set({ token, isAuthenticated: true });
        await get().fetchUser();
      }
    } catch {
      // Token invalid or expired
      await get().logout();
    } finally {
      set({ isLoading: false });
    }
  },

  fetchUser: async () => {
    try {
      const res = await api.get("/auth/me");
      set({ user: res.data });
    } catch {
      await get().logout();
    }
  },
}));

export default useAuthStore;
