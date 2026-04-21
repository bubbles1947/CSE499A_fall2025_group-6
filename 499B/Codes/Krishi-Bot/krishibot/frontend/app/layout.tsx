import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import ConditionalFooter from "@/components/ConditionalFooter";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "KrishiBot - AI Agriculture Assistant",
  description:
    "KrishiBot is an AI-powered agricultural assistant that helps farmers with crop disease detection, irrigation advice, fertilizer planning, and pest control — powered by a local LLM.",
  keywords: ["agriculture", "AI", "crop disease", "farming", "Bangladesh", "irrigation"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-primary-50 text-gray-900 antialiased flex flex-col min-h-screen">
        <Navbar />
        <main className="flex-1 flex flex-col">{children}</main>
        <ConditionalFooter />
      </body>
    </html>
  );
}
