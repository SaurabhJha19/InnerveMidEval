import 'package:flutter/material.dart';
import 'dart:async';
import 'main.dart'; // Replace with your main screen file

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  double _progress = 0.0;

  @override
  void initState() {
    super.initState();
    _startLoading();
  }

  void _startLoading() {
    Timer.periodic(Duration(milliseconds: 500), (Timer timer) {
      if (_progress >= 1.0) {
        timer.cancel();
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (context) => MyApp()), // Redirect to main screen
        );
      } else {
        setState(() {
          _progress += 0.2;
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white, // Change background color if needed
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset('assets/logo.png', width: 450, height: 450), // Logo
            SizedBox(height: 30),
            Text("Loading...", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 20),
            LinearProgressIndicator(value: _progress, minHeight: 2 ),
          ],
        ),
      ),
    );
  }
}
