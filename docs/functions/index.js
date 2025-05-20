const functions = require("firebase-functions");
const admin = require("firebase-admin");
const express = require("express");
const path = require("path");
const cookieParser = require("cookie-parser");

admin.initializeApp();

const app = express();
app.use(cookieParser());

// Dummy route to verify function is hit
app.get("/dummy-route", (req, res) => {
  res.send("Hello from Cloud Function!");
});

// Serve login.html (NOT protected, now from inside functions/static)
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
app.post("/sessionLogin", express.json(), async (req, res) => {
  const idToken = req.body.idToken;
  const expiresIn = 60 * 60 * 24 * 5 * 1000;
  try {
    const sessionCookie = await admin.auth().createSessionCookie(
        idToken,
        {maxAge: expiresIn, httpOnly: true, secure: true},
    );
    res.cookie(
        "__session",
        sessionCookie,
        {maxAge: expiresIn, httpOnly: true, secure: true},
    );
    res.status(200).send({status: "success"});
  } catch (error) {
    res.status(401).send("UNAUTHORIZED REQUEST!");
  }
});

// Auth middleware (protects docs and all other routes except above)
app.use(async (req, res, next) => {
  const openPaths = [
    "/login.html",
    "/sessionLogin",
    "/dummy-route",
  ];
  if (openPaths.includes(req.path)) {
    return next();
  }
  const sessionCookie = req.cookies.__session || "";
  try {
    await admin.auth().verifySessionCookie(sessionCookie, true);
    next();
  } catch (e) {
    res.redirect("/login.html");
  }
});

// Serve static Sphinx files (protected)
const DOCS_PATH = path.join(__dirname, "../_build/html");
app.use(express.static(DOCS_PATH));

exports.authCheck = functions.https.onRequest(app);
