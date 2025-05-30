import 'package:flutter/material.dart';
import 'basic_screen.dart';


class HomeScreen extends StatelessWidget{
  const HomeScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context){
    return SafeArea(
      top: true,
      bottom: false,
      child: BasicScreen(),
    );
  }
}
