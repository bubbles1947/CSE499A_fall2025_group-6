import React, { useEffect, useRef } from "react";
import { Animated, View } from "react-native";

/**
 * Animated three-dot typing indicator (WhatsApp-style bounce).
 */
export default function TypingIndicator() {
  const dot1 = useRef(new Animated.Value(0)).current;
  const dot2 = useRef(new Animated.Value(0)).current;
  const dot3 = useRef(new Animated.Value(0)).current;

  const bounce = (dot: Animated.Value, delay: number) =>
    Animated.loop(
      Animated.sequence([
        Animated.delay(delay),
        Animated.timing(dot, { toValue: -6, duration: 280, useNativeDriver: true }),
        Animated.timing(dot, { toValue: 0,  duration: 280, useNativeDriver: true }),
        Animated.delay(600),
      ])
    );

  useEffect(() => {
    const a1 = bounce(dot1, 0);
    const a2 = bounce(dot2, 160);
    const a3 = bounce(dot3, 320);
    a1.start(); a2.start(); a3.start();
    return () => { a1.stop(); a2.stop(); a3.stop(); };
  }, []);

  const dotStyle = (anim: Animated.Value) => ({
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: "#9CA3AF",
    marginHorizontal: 3,
    transform: [{ translateY: anim }],
  });

  return (
    <View
      style={{
        alignSelf: "flex-start",
        backgroundColor: "#2A2A3E",
        borderRadius: 18,
        borderBottomLeftRadius: 4,
        paddingHorizontal: 14,
        paddingVertical: 12,
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 8,
        marginLeft: 4,
      }}
    >
      <Animated.View style={dotStyle(dot1)} />
      <Animated.View style={dotStyle(dot2)} />
      <Animated.View style={dotStyle(dot3)} />
    </View>
  );
}
