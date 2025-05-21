const functions = require("firebase-functions");
const admin = require("firebase-admin");
const express = require("express");
const path = require("path");
const cookieParser = require("cookie-parser");

admin.initializeApp();

const app = express();
app.use(cookieParser());
app.use(express.json()); // Ensures req.body is parsed

// Dummy route to verify function is hit
app.get("/dummy-route", (req, res) => {
  res.send("Hello from Cloud Function!");
});

// Serve login.html (NOT protected, from inside functions/static)
app.get("/login.html", (req, res) => {
  const loginPath = path.join(__dirname, "static/login.html");
  console.log("Serving login.html from:", loginPath);
  res.sendFile(loginPath, (err) => {
    if (err) {
      console.error("Error sending login.html:", err);
      res.status(err.status || 500).end();
    }
  });
});

// Session login endpoint (NOT protected)
app.post("/sessionLogin", async (req, res) => {
  const idToken = req.body.idToken;
  const expiresIn = 60 * 60 * 24 * 5 * 1000; // 5 days

  if (!idToken) {
    res.status(400).send("No ID token provided");
    return;
  }

  try {
    // Optional: verify the token for better error messages
    const decoded = await admin.auth().verifyIdToken(idToken);
    console.log("ID token decoded:", decoded);

    const sessionCookie = await admin.auth().createSessionCookie(
        idToken, {expiresIn},
    );

    res.cookie(
        "__session",
        sessionCookie,
        {
          maxAge: expiresIn,
          httpOnly: true,
          secure: true, // For local testing, use false or set up HTTPS
          path: "/",
        },
    );
    res.status(200).send({status: "success"});
  } catch (error) {
    console.error("Session cookie creation failed:", error);
    res.status(401).send("UNAUTHORIZED REQUEST!");
  }
});

// Logout endpoint (optional but recommended)
app.post("/logout", (req, res) => {
  res.clearCookie("__session", {path: "/"});
  res.status(200).send({status: "logged out"});
});

// Auth middleware (protects docs and all other routes except above)
app.use(async (req, res, next) => {
  const openPaths = [
    "/login.html",
    "/sessionLogin",
    "/dummy-route",
    "/logout",
  ];

  // Allow any open path or paths that start with them (for query params)
  if (openPaths.some((p) => req.path.startsWith(p))) {
    return next();
  }
  console.log("Session cookie raw value:", req.cookies.__session);
  try {
    await admin.auth().verifySessionCookie(req.cookies.__session, true);
    next();
  } catch (e) {
    console.error("Session verification failed:", e);
    res.redirect("/login.html");
  }
});

// Serve static Sphinx files (protected)
const DOCS_PATH = path.join(__dirname, "_build/html");
app.use(express.static(DOCS_PATH));

// Export the cloud function
exports.authCheck = functions.https.onRequest(app);
