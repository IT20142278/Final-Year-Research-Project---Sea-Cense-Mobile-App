import 'dart:async';

import 'package:flutter/material.dart';
import 'package:sea_cense/views/onboarding/info_view.dart';
import 'package:shimmer_animation/shimmer_animation.dart';

class SplashView extends StatefulWidget {
  const SplashView({super.key});

  @override
  State<SplashView> createState() => _SplashViewState();
}

class _SplashViewState extends State<SplashView> {
  @override
  void initState() {
    Timer(const Duration(seconds: 4), () {
      Navigator.of(context).push(MaterialPageRoute(builder: (context) => const InfoView()));
    });
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        height: MediaQuery.of(context).size.height,
        width: MediaQuery.of(context).size.width,
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/splash_bg.png"),
            fit: BoxFit.cover,
          ),
        ),
        child: Shimmer(
          duration: const Duration(seconds: 1),
          interval: const Duration(seconds: 1),
          color: const Color(0xFF0A78C2),
          colorOpacity: 0.1,
          child: Image.asset('assets/images/splash_logo.png'),
        ),
      ),
    );
  }
}
