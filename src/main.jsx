import React from "react";
import { createRoot } from "react-dom/client";
import RLUsersDashboard from "../visuals/dashboard.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RLUsersDashboard />
  </React.StrictMode>
);
