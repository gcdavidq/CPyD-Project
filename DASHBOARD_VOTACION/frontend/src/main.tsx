import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./estilos/global.css";
import "./componentes/Conmutador.css";

const raiz = document.getElementById("raiz");
if (!raiz) throw new Error("No se encontró el contenedor #raiz");

createRoot(raiz).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
