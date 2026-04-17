import React, { useEffect } from "react";
import { StatusBar } from "expo-status-bar";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import "./global.css";

import AppNavigator from "./src/navigation/AppNavigator";
import useAuthStore from "./src/store/useAuthStore";

export default function App() {
  const loadToken = useAuthStore((s) => s.loadToken);

  // On app launch, try to restore the saved JWT token
  useEffect(() => {
    loadToken();
  }, []);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <StatusBar style="light" />
        <AppNavigator />
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}
