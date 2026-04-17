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
  navigation: NativeStackNavigationProp<AuthStackParamList, "Register">;
};

export default function RegisterScreen({ navigation }: Props) {
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const register = useAuthStore((s) => s.register);

  const handleRegister = async () => {
    if (!email.trim() || !username.trim() || !password || !confirmPassword) {
      Alert.alert("Error", "Please fill in all fields.");
      return;
    }
    if (password !== confirmPassword) {
      Alert.alert("Error", "Passwords do not match.");
      return;
    }
    if (password.length < 8) {
      Alert.alert("Error", "Password must be at least 8 characters.");
      return;
    }

    setLoading(true);
    try {
      const cleanEmail = email.trim().toLowerCase();
      const cleanUsername = username.trim();
      const cleanPassword = password.trim();
      console.log("[Register] Request →", { email: cleanEmail, username: cleanUsername });
      await register(cleanEmail, cleanUsername, cleanPassword);
      console.log("[Register] Success ✓");
    } catch (err: any) {
      console.log("[Register] Error ✗", err.message, err.response?.data);
      Alert.alert("Registration Failed", err.message || "Registration failed. Please try again.");
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
        {/* Title */}
        <View className="items-center mb-10">
          <Ionicons name="person-add" size={56} color="#6C63FF" />
          <Text className="text-white text-2xl font-bold mt-4">
            Create Account
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

        {/* Username */}
        <View className="bg-dark-100 rounded-xl px-4 py-3 mb-4 flex-row items-center">
          <Ionicons name="person-outline" size={20} color="#888" />
          <TextInput
            className="flex-1 text-white ml-3 text-base"
            placeholder="Username"
            placeholderTextColor="#666"
            autoCapitalize="none"
            value={username}
            onChangeText={setUsername}
          />
        </View>

        {/* Password */}
        <View className="bg-dark-100 rounded-xl px-4 py-3 mb-4 flex-row items-center">
          <Ionicons name="lock-closed-outline" size={20} color="#888" />
          <TextInput
            className="flex-1 text-white ml-3 text-base"
            placeholder="Password (min 8 characters)"
            placeholderTextColor="#666"
            secureTextEntry
            autoCapitalize="none"
            autoCorrect={false}
            spellCheck={false}
            value={password}
            onChangeText={setPassword}
          />
        </View>

        {/* Confirm Password */}
        <View className="bg-dark-100 rounded-xl px-4 py-3 mb-6 flex-row items-center">
          <Ionicons name="shield-checkmark-outline" size={20} color="#888" />
          <TextInput
            className="flex-1 text-white ml-3 text-base"
            placeholder="Confirm Password"
            placeholderTextColor="#666"
            secureTextEntry
            autoCapitalize="none"
            autoCorrect={false}
            spellCheck={false}
            value={confirmPassword}
            onChangeText={setConfirmPassword}
          />
        </View>

        {/* Register Button */}
        <TouchableOpacity
          className="bg-primary rounded-xl py-4 items-center mb-4"
          onPress={handleRegister}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text className="text-white text-lg font-semibold">Sign Up</Text>
          )}
        </TouchableOpacity>

        {/* Back to Login */}
        <TouchableOpacity
          className="items-center py-2"
          onPress={() => navigation.goBack()}
        >
          <Text className="text-gray-400">
            Already have an account?{" "}
            <Text className="text-primary font-semibold">Sign In</Text>
          </Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}
