import React from "react";
import { ActivityIndicator, View } from "react-native";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { Ionicons } from "@expo/vector-icons";

import useAuthStore from "../store/useAuthStore";
import LoginScreen from "../screens/LoginScreen";
import RegisterScreen from "../screens/RegisterScreen";
import HomeScreen from "../screens/HomeScreen";
import ChatScreen from "../screens/ChatScreen";
import DocumentScreen from "../screens/DocumentScreen";

// ── Type definitions ──────────────────────────────────────────────

export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
};

// Chat is accessed from the root stack so it gets a back button
export type RootStackParamList = {
  MainTabs: undefined;
  Chat: { conversationId?: string; documentId?: string; documentName?: string } | undefined;
};

export type MainTabParamList = {
  Home: undefined;
  Documents: undefined;
};

const AuthStack = createNativeStackNavigator<AuthStackParamList>();
const RootStack  = createNativeStackNavigator<RootStackParamList>();
const MainTab    = createBottomTabNavigator<MainTabParamList>();

// ── Auth Navigator ────────────────────────────────────────────────

function AuthNavigator() {
  return (
    <AuthStack.Navigator
      screenOptions={{
        headerShown: false,
        contentStyle: { backgroundColor: "#15152A" },
      }}
    >
      <AuthStack.Screen name="Login"    component={LoginScreen}    />
      <AuthStack.Screen name="Register" component={RegisterScreen} />
    </AuthStack.Navigator>
  );
}

// ── Main Tab Navigator (Home + Documents only) ────────────────────

function MainTabNavigator() {
  return (
    <MainTab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarStyle: {
          backgroundColor: "#1E1E2E",
          borderTopColor: "#2A2A3E",
          borderTopWidth: 1,
          height: 62,
          paddingBottom: 10,
          paddingTop: 8,
        },
        tabBarActiveTintColor: "#7C3AED",
        tabBarInactiveTintColor: "#666",
        tabBarLabelStyle: { fontSize: 11 },
        tabBarIcon: ({ color, size }) => {
          const icon: keyof typeof Ionicons.glyphMap =
            route.name === "Home" ? "home" : "document-text";
          return <Ionicons name={icon} size={size - 2} color={color} />;
        },
      })}
    >
      <MainTab.Screen name="Home"      component={HomeScreen}     />
      <MainTab.Screen name="Documents" component={DocumentScreen} />
    </MainTab.Navigator>
  );
}

// ── Root Navigator (wraps tabs + full-screen Chat) ────────────────

function RootNavigator() {
  return (
    <RootStack.Navigator screenOptions={{ headerShown: false }}>
      <RootStack.Screen name="MainTabs" component={MainTabNavigator} />
      <RootStack.Screen
        name="Chat"
        component={ChatScreen}
        options={{
          animation: "slide_from_right",
          contentStyle: { backgroundColor: "#15152A" },
        }}
      />
    </RootStack.Navigator>
  );
}

// ── App Entry ─────────────────────────────────────────────────────

export default function AppNavigator() {
  const { isAuthenticated, isLoading } = useAuthStore();

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#15152A" }}>
        <ActivityIndicator size="large" color="#7C3AED" />
      </View>
    );
  }

  return (
    <NavigationContainer>
      {isAuthenticated ? <RootNavigator /> : <AuthNavigator />}
    </NavigationContainer>
  );
}
