/**
 * Electron main process for Dental 3D capture client.
 *
 * In development: loads the Vite dev server (http://localhost:5173).
 * In production: loads the built dist/index.html.
 *
 * UVC camera control:
 *   Standard intraoral UVC cameras are exposed to the renderer via
 *   navigator.mediaDevices (WebRTC) — sufficient for POC.
 *   Production UVC extension units (exposure lock, white balance lock)
 *   are accessible via the `usb` npm package; wire through IPC when needed.
 */
const { app, BrowserWindow, ipcMain, shell } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

const isDev = process.env.NODE_ENV === "development" || !app.isPackaged;

let mainWin = null;

function createWindow() {
  mainWin = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    title: "Dental 3D",
    backgroundColor: "#0f1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      // Allow camera access without HTTPS when running locally
      webSecurity: !isDev,
    },
  });

  if (isDev) {
    mainWin.loadURL("http://localhost:5173");
    mainWin.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWin.loadFile(path.join(__dirname, "../dist/index.html"));
  }

  mainWin.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
}

app.whenReady().then(() => {
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

// ── IPC: camera permission helper ────────────────────────────────────────────
// On macOS, camera permission must be requested from the main process.
ipcMain.handle("request-camera-permission", async () => {
  if (process.platform !== "darwin") return "granted";
  const { systemPreferences } = require("electron");
  const status = systemPreferences.getMediaAccessStatus("camera");
  if (status === "not-determined") {
    return systemPreferences.askForMediaAccess("camera");
  }
  return status;
});

// ── IPC: open file in OS default app (for OBJ export) ────────────────────────
ipcMain.handle("open-file", (_, filePath) => shell.openPath(filePath));

// ── IPC: get app version ──────────────────────────────────────────────────────
ipcMain.handle("get-version", () => app.getVersion());
