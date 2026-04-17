import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from "react-native";

import { NativeStackNavigationProp } from "@react-navigation/native-stack";
import { Ionicons } from "@expo/vector-icons";
import useAuthStore from "../store/useAuthStore";
import { AuthStackParamList } from "../navigation/AppNavigator";

type Props = {
  navigation: NativeStackNavigationProp<AuthStackParamList, "Login">;
};

export default function LoginScreen({ navigation }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const login = useAuthStore((s) => s.login);

  const handleLogin = async () => {
    setError("");
    if (!email.trim() || !password.trim()) {
      setError("Please fill in all fields.");
      return;
    }
    setLoading(true);
    try {
      const cleanEmail = email.trim().toLowerCase();
      const cleanPassword = password.trim();
      console.log("[Login] Request →", { email: cleanEmail });
      await login(cleanEmail, cleanPassword);
      console.log("[Login] Success ✓");
    } catch (err: any) {
      const msg = err.message || "Login failed. Check your credentials.";
      console.log("[Login] Error ✗", msg, err.response?.data);
      setError(msg);
      // Also show native alert on mobile for visibility
      if (Platform.OS !== "web") {
        Alert.alert("Login Failed", msg);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      className="flex-1 bg-dark-300"
    >
      <View className="flex-1 justify-center px-8">
        {/* Logo / Title */}
        <View className="items-center mb-10">
          <Ionicons name="hardware-chip" size={64} color="#6C63FF" />
          <Text className="text-white text-3xl font-bold mt-4">LocalMind AI</Text>
          <Text className="text-gray-400 text-base mt-2">
            100% Local AI — Zero Cloud
          </Text>
        </View>

        {/* Email */}
        <View className="bg-dark-100 rounded-xl px-4 py-3 mb-4 flex-row items-center">
          <Ionicons name="mail-outline" size={20} color="#888" />
          <TextInput
            className="flex-1 text-white ml-3 text-base"
            placeholder="Email"
            placeholderTextColor="#666"
            keyboardType="email-address"
            autoCapitalize="none"
            value={email}
            onChangeText={setEmail}
          />
        </View>

        {/* Password */}
        <View className="bg-dark-100 rounded-xl px-4 py-3 mb-6 flex-row items-center">
          <Ionicons name="lock-closed-outline" size={20} color="#888" />
          <TextInput
            className="flex-1 text-white ml-3 text-base"
            placeholder="Password"
            placeholderTextColor="#666"
            secureTextEntry
            autoCapitalize="none"
            autoCorrect={false}
            spellCheck={false}
            value={password}
            onChangeText={setPassword}
          />
        </View>

        {/* Inline error — visible on both web and mobile */}
        {error ? (
          <Text style={{ color: "#F87171", fontSize: 13, marginBottom: 12, textAlign: "center" }}>
            {error}
          </Text>
        ) : null}

        {/* Login Button */}
        <TouchableOpacity
          className="bg-primary rounded-xl py-4 items-center mb-4"
          onPress={handleLogin}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text className="text-white text-lg font-semibold">Sign In</Text>
          )}
        </TouchableOpacity>

        {/* Register Link */}
        <TouchableOpacity
          className="items-center py-2"
          onPress={() => navigation.navigate("Register")}
        >
          <Text className="text-gray-400">
            Don't have an account?{" "}
            <Text className="text-primary font-semibold">Sign Up</Text>
          </Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}
