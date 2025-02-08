import 'package:flutter/material.dart';
import 'splash_screen.dart';
import 'call_log_page.dart';


void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    home: SplashScreen(), // Start with the splash screen
  ));
}


class MyApp extends StatelessWidget {
  const MyApp({super.key});
@override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[200], // Light background color
      body: SafeArea(
        child: Column(
          children: [
            // Header Section
            Container(
              padding: EdgeInsets.symmetric(horizontal: 12, vertical: 1),
              width: double.infinity,
              color: Colors.white, // White background for header
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center, // Center items horizontally
                crossAxisAlignment: CrossAxisAlignment.center, // Align items vertically
                children: [
                  Image.asset('assets/header.png', width: 80, height: 80), // Logo
                  SizedBox(width: 10),
                  Text(
                    "TRUEVISAGE", // App name
                    style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),

            // Space below header
            SizedBox(height: 20),
            
            // Call Log Button
            ElevatedButton.icon(
              icon: Icon(Icons.call),
              label: Text("View Call Log"),
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(horizontal: 50, vertical: 50),
                textStyle: TextStyle(fontSize: 18),
              ),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => CallLogScreen()),
                );
              },
            ),
            // Rest of the content will go here
          ],
        ),
      ),
    );
  }
}

