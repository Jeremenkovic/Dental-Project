/**
 * Preload script: exposes a minimal, typed API from main → renderer.
 * contextIsolation = true means the renderer cannot access Node directly.
 */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("dental", {
  requestCameraPermission: () => ipcRenderer.invoke("request-camera-permission"),
  openFile: (path) => ipcRenderer.invoke("open-file", path),
  getVersion: () => ipcRenderer.invoke("get-version"),
  isElectron: true,
});
