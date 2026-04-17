import axios from "axios";
import { Platform } from "react-native";

// Web browser (Expo web) must hit localhost; phone uses the PC's WiFi IP
const API_BASE_URL =
  Platform.OS === "web"
    ? "http://localhost:8000"
    : "http://192.168.0.102:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// Response interceptor — normalize error messages for the UI
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const detail = error.response?.data?.detail;

    // FastAPI validation errors come as an array: [{msg: "..."}]
    if (Array.isArray(detail)) {
      error.message = detail.map((d: any) => d.msg).join(". ");
    } else if (typeof detail === "string") {
      error.message = detail;
    } else if (!error.response) {
      error.message = "Cannot reach server. Make sure you are on the same WiFi as the PC.";
    }

    return Promise.reject(error);
  }
);

export default api;
